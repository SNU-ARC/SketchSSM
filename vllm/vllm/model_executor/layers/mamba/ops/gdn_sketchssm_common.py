# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM storage, q/k rotation and cold sketch build for Gated DeltaNet.

The state ``(slot, HV, V, K)`` is kept in rotated key coordinates. Per
request, a value head of rank ``m`` (0 = dense) keeps packed BF16 rows ``u``
(leading key columns), ``phi`` (coefficient map) and ``fs`` (projected erase
history). Dense heads keep all ``K`` key columns in ``u``: non-flush steps
read them instead of the FP32 state, which changes only at flushes. Their
layout ``u_off`` is -1 when they have no rows (ReplaySSM: every head dense,
the FP32 state read at every step). Sketch buffers are indexed by the
persistent request index.
"""

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

GDN_SKETCH_PIVOTS = 4
# Default window; the kernels take any multiple of GDN_SKETCH_WINDOW_ALIGN.
GDN_SKETCH_WINDOW = 16
GDN_SKETCH_WINDOW_ALIGN = 16
GDN_SKETCH_HEAD_DIM = 128


def gdn_sketch_window_supported(window: int) -> bool:
    """Whether the GDN SketchSSM kernels take this window length."""
    return window >= GDN_SKETCH_WINDOW_ALIGN and window % GDN_SKETCH_WINDOW_ALIGN == 0


def gdn_sketch_layout(
    ranks: torch.Tensor, window: int = GDN_SKETCH_WINDOW, dense_rows: bool = True
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Per-head ``(u_off, phi_off, fs_off, FG)`` table and packed row lengths.

    With ``dense_rows``, dense heads keep their ``K`` BF16 key columns in
    ``u``; otherwise their ``u_off`` is -1.
    """
    if not gdn_sketch_window_supported(window):
        raise ValueError(
            f"GDN SketchSSM window {window} is not a multiple of "
            f"{GDN_SKETCH_WINDOW_ALIGN}"
        )
    k, p = GDN_SKETCH_HEAD_DIM, GDN_SKETCH_PIVOTS
    rows = []
    nu = nm = nf = 0
    for m in ranks.tolist():
        if not 0 <= m <= k:
            raise ValueError(f"GDN sketch rank {m} outside [0, {k}]")
        fg = max(4, (m + 3) // 4 * 4)
        rows.append((nu if m or dense_rows else -1, nm, nf, fg))
        nu += m * k if m else k * k if dense_rows else 0
        nm += 0 if m in (0, k) else m * k if m <= p else p * k + (p + 1) * fg
        nf += window * fg if m else 0
    # Rows are staged with 16-byte copies.
    sizes = tuple(max(8, (n + 7) // 8 * 8) for n in (nu, nm, nf))
    return torch.tensor(rows, dtype=torch.int32), sizes


def gdn_rank_cap(ranks: torch.Tensor) -> int:
    """Rank bound ``G`` the kernels are specialized on (multiple of 8)."""
    return max(8, (int(ranks.max()) + 7) // 8 * 8)


class GDNSketchTables(torch.nn.Module):
    """Ranks and packed-layout table of one GDN layer (``dense_rows``: see
    ``gdn_sketch_layout``)."""

    def __init__(
        self,
        ranks: torch.Tensor,
        num_k_heads: int,
        window: int = GDN_SKETCH_WINDOW,
        dense_rows: bool = True,
    ):
        super().__init__()
        ranks = ranks.to(torch.int32).cpu()
        layout, self.sizes = gdn_sketch_layout(ranks, window, dense_rows)
        self.num_k_heads = num_k_heads
        self.window = window
        self.num_v_heads = ranks.numel()
        self.rank_cap = gdn_rank_cap(ranks)
        device = torch.get_default_device()
        self.register_buffer("ranks", ranks.to(device), persistent=False)
        self.register_buffer("layout", layout.to(device), persistent=False)

    def buffer_specs(self) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """Per-request ``name: (shape, dtype)`` of the sketch buffers."""
        nu, nm, nf = self.sizes
        hv, h = self.num_v_heads, self.num_k_heads
        d, w = GDN_SKETCH_HEAD_DIM, self.window
        return {
            "u": ((nu,), torch.bfloat16),
            "phi": ((nm,), torch.bfloat16),
            "fs": ((nf,), torch.bfloat16),
            "beta": ((hv, w), torch.float32),
            "current_d": ((hv, d), torch.float32),
            "current_k": ((h, d), torch.float32),
        }


@dataclass
class GDNSketchArgs:
    """One GDN layer's per-request sketch buffers (first dim: request index)."""

    u: torch.Tensor
    phi: torch.Tensor
    fs: torch.Tensor
    beta: torch.Tensor
    current_d: torch.Tensor
    current_k: torch.Tensor
    tables: GDNSketchTables

    @classmethod
    def allocate(
        cls, tables: GDNSketchTables, max_num_reqs: int, device=None
    ) -> "GDNSketchArgs":
        buffers = {
            name: torch.zeros(max_num_reqs, *shape, dtype=dtype, device=device)
            for name, (shape, dtype) in tables.buffer_specs().items()
        }
        return cls(**buffers, tables=tables)


@triton.jit(do_not_specialize=["T"])
def _rotate_qk_kernel(X, RT, T, s_t, HK: tl.constexpr, K: tl.constexpr,
                      BT: tl.constexpr):  # fmt: skip
    # In place, q then k of key head h: x <- R x.
    pt = tl.program_id(0)
    h = tl.program_id(1)
    rows = pt * BT + tl.arange(0, BT)
    cols = tl.arange(0, K)
    rt = tl.load(RT + h * K * K + cols[:, None] * K + cols[None, :])
    for z in tl.static_range(2):
        ptr = X + rows[:, None].to(tl.int64) * s_t + (z * HK + h) * K + cols[None, :]
        x = tl.load(ptr, mask=rows[:, None] < T, other=0.0).to(tl.float32)
        y = tl.dot(x, rt, input_precision="tf32x3")
        tl.store(ptr, y.to(X.dtype.element_ty), mask=rows[:, None] < T)


def gdn_rotation_from_frames(frames: torch.Tensor) -> torch.Tensor:
    """``(H, K, K)`` FP32 ``R^T`` of a layer's per-key-head frames."""
    return frames.to(torch.float32).transpose(-1, -2).contiguous()


def gdn_sketch_rotate_(mixed_qkv: torch.Tensor, frames_t: torch.Tensor) -> None:
    """Rotate q and k of ``mixed_qkv (tokens, 2 H K + HV V)`` in place.

    ``frames_t`` is ``gdn_rotation_from_frames(frames)``.
    """
    num_tokens = mixed_qkv.shape[0]
    if num_tokens == 0:
        return
    h, k, _ = frames_t.shape
    assert mixed_qkv.stride(1) == 1 and frames_t.is_contiguous()
    assert frames_t.dtype == torch.float32
    bt = 64 if num_tokens >= 1024 else 32
    _rotate_qk_kernel[(triton.cdiv(num_tokens, bt), h)](
        mixed_qkv, frames_t, num_tokens, mixed_qkv.stride(0), HK=h, K=k, BT=bt,
        num_warps=4,
    )  # fmt: skip


@triton.jit
def _build_head(base, sk, h, u, packed, fs, beta, widths, layout, SU: tl.constexpr,
                SM: tl.constexpr, SF: tl.constexpr, SV: tl.constexpr,
                SK: tl.constexpr, HV: tl.constexpr, G: tl.constexpr,
                K: tl.constexpr, V: tl.constexpr, W: tl.constexpr):  # fmt: skip
    # Builds one head's sketch and resets its erase history and gates.
    j = tl.arange(0, K)
    v = tl.arange(0, V)
    w = tl.arange(0, 16)
    sk = sk.to(tl.int64)
    for wr in tl.static_range(0, W, 16):
        tl.store(beta + (sk * HV + h) * W + wr + w, tl.zeros([16], tl.float32))
    m = tl.load(widths + h)
    uoff = tl.load(layout + h * 4)
    if (m == 0) & (uoff >= 0):
        # Dense head: all K key columns, read on the non-flush steps.
        s = tl.load(base + v[:, None] * SV + j[None, :] * SK)
        tl.store(
            u + sk * SU + uoff + j[None, :] * V + v[:, None],
            s.to(u.dtype.element_ty),
        )
    if m > 0:
        moff = tl.load(layout + h * 4 + 1)
        foff = tl.load(layout + h * 4 + 2)
        fg = tl.load(layout + h * 4 + 3)
        o = tl.arange(0, 16 * K)
        for wr in tl.static_range(0, W, 16):
            tl.store(
                fs + sk * SF + foff + wr * fg + o,
                tl.zeros([16 * K], fs.dtype.element_ty),
                o < 16 * fg,
            )
        s = tl.load(base + v[:, None] * SV + j[None, :] * SK).to(tl.float32)
        tl.store(
            u + sk * SU + uoff + j[None, :] * V + v[:, None],
            s,
            (j[None, :] < m) & (j[None, :] < G),
        )
        if m < K:
            energy = tl.sum(s * s, axis=0)
            mean = tl.sum(energy) / K
            safe_mean = tl.where(mean > 0.0, mean, 1.0)
            u0 = tl.load(base + v * SV + 0 * SK).to(tl.float32)
            v0 = tl.full((V,), 0.0, tl.float32)
            if m > 0:
                w0 = u0
                e0 = tl.sum(w0 * w0)
                keep0 = e0 > 0.0
                v0 = tl.where(keep0, w0 / tl.sqrt(tl.where(keep0, e0, 1.0)), 0.0)
            z0 = tl.sum(v0[:, None] * s, axis=0) / tl.sqrt(safe_mean)
            u1 = tl.load(base + v * SV + 1 * SK).to(tl.float32)
            v1 = tl.full((V,), 0.0, tl.float32)
            if m > 1:
                w1 = u1
                w1 = w1 - v0 * tl.sum(v0 * w1)
                w1 = w1 - v0 * tl.sum(v0 * w1)
                e1 = tl.sum(w1 * w1)
                keep1 = e1 > 1.0e-12 * tl.sum(u1 * u1)
                v1 = tl.where(keep1, w1 / tl.sqrt(tl.where(keep1, e1, 1.0)), 0.0)
            z1 = tl.sum(v1[:, None] * s, axis=0) / tl.sqrt(safe_mean)
            u2 = tl.load(base + v * SV + 2 * SK).to(tl.float32)
            v2 = tl.full((V,), 0.0, tl.float32)
            if m > 2:
                w2 = u2
                w2 = w2 - v0 * tl.sum(v0 * w2)
                w2 = w2 - v1 * tl.sum(v1 * w2)
                w2 = w2 - v0 * tl.sum(v0 * w2)
                w2 = w2 - v1 * tl.sum(v1 * w2)
                e2 = tl.sum(w2 * w2)
                keep2 = e2 > 1.0e-12 * tl.sum(u2 * u2)
                v2 = tl.where(keep2, w2 / tl.sqrt(tl.where(keep2, e2, 1.0)), 0.0)
            z2 = tl.sum(v2[:, None] * s, axis=0) / tl.sqrt(safe_mean)
            u3 = tl.load(base + v * SV + 3 * SK).to(tl.float32)
            v3 = tl.full((V,), 0.0, tl.float32)
            if m > 3:
                w3 = u3
                w3 = w3 - v0 * tl.sum(v0 * w3)
                w3 = w3 - v1 * tl.sum(v1 * w3)
                w3 = w3 - v2 * tl.sum(v2 * w3)
                w3 = w3 - v0 * tl.sum(v0 * w3)
                w3 = w3 - v1 * tl.sum(v1 * w3)
                w3 = w3 - v2 * tl.sum(v2 * w3)
                e3 = tl.sum(w3 * w3)
                keep3 = e3 > 1.0e-12 * tl.sum(u3 * u3)
                v3 = tl.where(keep3, w3 / tl.sqrt(tl.where(keep3, e3, 1.0)), 0.0)
            z3 = tl.sum(v3[:, None] * s, axis=0) / tl.sqrt(safe_mean)
            residual = tl.maximum(
                energy / safe_mean - z0 * z0 - z1 * z1 - z2 * z2 - z3 * z3, 0.0
            )
            residual = tl.where(j < tl.minimum(m, 4), 0.0, residual)
            denominator = residual + 0.1
            a = residual / denominator
            b0 = tl.where(j < m, z0 / denominator, 0.0)
            b1 = tl.where(j < m, z1 / denominator, 0.0)
            b2 = tl.where(j < m, z2 / denominator, 0.0)
            b3 = tl.where(j < m, z3 / denominator, 0.0)
            l0_0 = tl.sqrt(1.0 + tl.sum(z0 * b0))
            l1_0 = (tl.sum(z1 * b0)) / l0_0
            l1_1 = tl.sqrt(1.0 + tl.sum(z1 * b1) - l1_0 * l1_0)
            l2_0 = (tl.sum(z2 * b0)) / l0_0
            l2_1 = (tl.sum(z2 * b1) - l2_0 * l1_0) / l1_1
            l2_2 = tl.sqrt(1.0 + tl.sum(z2 * b2) - l2_0 * l2_0 - l2_1 * l2_1)
            l3_0 = (tl.sum(z3 * b0)) / l0_0
            l3_1 = (tl.sum(z3 * b1) - l3_0 * l1_0) / l1_1
            l3_2 = (tl.sum(z3 * b2) - l3_0 * l2_0 - l3_1 * l2_1) / l2_2
            l3_3 = tl.sqrt(
                1.0 + tl.sum(z3 * b3) - l3_0 * l3_0 - l3_1 * l3_1 - l3_2 * l3_2
            )
            y0 = (b0) / l0_0
            y1 = (b1 - l1_0 * y0) / l1_1
            y2 = (b2 - l2_0 * y0 - l2_1 * y1) / l2_2
            y3 = (b3 - l3_0 * y0 - l3_1 * y1 - l3_2 * y2) / l3_3
            g3 = (y3) / l3_3
            g2 = (y2 - l3_2 * g3) / l2_2
            g1 = (y1 - l2_1 * g2 - l3_1 * g3) / l1_1
            g0 = (y0 - l1_0 * g1 - l2_0 * g2 - l3_0 * g3) / l0_0
            factor = tl.where(j < m, 0.1 / denominator, 1.0)
            tbase = packed + sk * SM + moff + j
            index = sk * SM + moff + 4 * K + j
            if m <= 4:
                for n in range(m):
                    f0 = tl.sum(tl.where(j == n, g0, 0.0))
                    f1 = tl.sum(tl.where(j == n, g1, 0.0))
                    f2 = tl.sum(tl.where(j == n, g2, 0.0))
                    f3 = tl.sum(tl.where(j == n, g3, 0.0))
                    merged = (f0 * z0 + f1 * z1 + f2 * z2 + f3 * z3) * factor
                    tl.store(tbase + n * K, merged)
            else:
                tl.store(packed + index, a, j < m)
                tl.store(tbase + 0 * K, z0 * factor)
                tl.store(packed + index + 1 * fg, g0, j < m)
                tl.store(tbase + 1 * K, z1 * factor)
                tl.store(packed + index + 2 * fg, g1, j < m)
                tl.store(tbase + 2 * K, z2 * factor)
                tl.store(packed + index + 3 * fg, g2, j < m)
                tl.store(tbase + 3 * K, z3 * factor)
                tl.store(packed + index + 4 * fg, g3, j < m)


@triton.jit
def _gdn_build_kernel(state, u, packed, fs, beta, widths, layout, slots, metas,
                      flags, rows, batch, null_block_id, SB: tl.constexpr,
                      SH: tl.constexpr, SV: tl.constexpr, SK: tl.constexpr,
                      SU: tl.constexpr, SM: tl.constexpr, SF: tl.constexpr,
                      HV: tl.constexpr, G: tl.constexpr, K: tl.constexpr,
                      V: tl.constexpr, W: tl.constexpr,
                      ROW_LIST: tl.constexpr):  # fmt: skip
    # With ROW_LIST, programs walk ``rows`` and stop at the first -1 padding.
    h = tl.program_id(1)
    it = tl.program_id(0)
    if ROW_LIST:
        row = tl.load(rows + it, mask=it < batch, other=-1)
    else:
        row = tl.where(tl.load(flags + it) != 0, it, -1)
    while row >= 0:
        slot = tl.load(slots + row).to(tl.int32)
        if (slot != null_block_id) & (slot >= 0):
            sk = tl.load(metas + row)
            _build_head(state + slot.to(tl.int64) * SB + h.to(tl.int64) * SH, sk, h,
                        u, packed, fs, beta, widths, layout, SU, SM, SF, SV, SK,
                        HV, G, K, V, W)  # fmt: skip
        if ROW_LIST:
            it += tl.num_programs(0)
            row = tl.load(rows + it, mask=it < batch, other=-1)
        else:
            row = row * 0 - 1


def gdn_sketch_build(
    state: torch.Tensor,
    flags: torch.Tensor,
    slots: torch.Tensor,
    meta: torch.Tensor,
    sketch: GDNSketchArgs,
    null_block_id: int = NULL_BLOCK_ID,
    rows: torch.Tensor | None = None,
    rows_per_program: int = 4,
) -> None:
    """Build the sketch of every flagged row from its state.

    ``rows`` is an optional -1 padded list of the flagged rows.
    """
    batch = flags.shape[0]
    if batch == 0:
        return
    t = sketch.tables
    _, hv, v, k = state.shape
    assert (v, k) == (GDN_SKETCH_HEAD_DIM, GDN_SKETCH_HEAD_DIM)
    assert slots.dtype == torch.int32 and meta.dtype == torch.int32
    programs = batch if rows is None else triton.cdiv(batch, rows_per_program)
    _gdn_build_kernel[(programs, hv)](
        state, sketch.u, sketch.phi, sketch.fs, sketch.beta, t.ranks, t.layout,
        slots, meta, flags, flags if rows is None else rows, batch, null_block_id,
        SB=state.stride(0), SH=state.stride(1), SV=state.stride(2),
        SK=state.stride(3), SU=sketch.u.stride(0), SM=sketch.phi.stride(0),
        SF=sketch.fs.stride(0), HV=hv, G=t.rank_cap, K=k, V=v,
        W=t.window, ROW_LIST=rows is not None, num_warps=4,
    )  # fmt: skip
