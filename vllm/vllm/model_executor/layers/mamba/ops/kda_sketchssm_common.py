# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM storage for Kimi Delta Attention (KDA).

KDA's per-channel decay does not commute with a rotation, so the state stays
in native coordinates and each head's frame is applied inside the kernels.
Per-request sketches are indexed by the persistent request index; the window
rings live next to the state in the Mamba cache page. For ``W > 16`` the
``d_ring`` rows of 8-row chunk ``c >= 1`` are rebased to ``prefix[8 c - 1]``
to keep their exponents bounded.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields

import torch

from vllm.triton_utils import tl, triton

KDA_SKETCH_PIVOTS = 4
KDA_SKETCH_WINDOW = 16
KDA_SKETCH_HEAD_DIM = 128
KDA_SKETCH_LOWER_BOUND = -5.0
# Rank buckets of the flush launches (one launch per non-empty bucket).
KDA_SKETCH_BUCKETS = ((0, 16), (16, 32), (32, 64), (64, 128))
# Floats of per-(row, sketch head) scratch between the flush and its finish.
KDA_SKETCH_SCRATCH_F = 1792
# Packed frame floats per head: FP32 frame, BF16 hi/lo columns, pivot rows.
_FRAME_F = 34880


def kda_sketch_window_supported(window: int) -> bool:
    """Windows the KDA sketch kernels support: positive multiples of 16."""
    return window >= 16 and window % 16 == 0


def _check_window(window: int) -> None:
    if not kda_sketch_window_supported(window):
        raise ValueError(f"KDA sketch window must be a multiple of 16, got {window}")


def kda_rank_cap(ranks: torch.Tensor) -> int:
    """Rank bound ``G`` the kernels are specialized on (multiple of 8)."""
    return max(8, (int(ranks.max()) + 7) // 8 * 8)


def _pack_frames(frames: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
    # Original frame, BF16 hi/lo in the kernel's permuted key order, then
    # natural-order TF32 hi/lo pivot rows.
    heads = frames.shape[0]
    device = frames.device
    original = frames.float().contiguous()
    packed = torch.zeros(heads, _FRAME_F, device=device, dtype=torch.float32)
    packed[:, :16384].copy_(original.reshape(heads, 16384))
    idx = torch.arange(128, device=device)
    p = idx % 16
    key = (idx // 16) * 16 + 4 * ((p & 7) >> 1) + (p & 1)
    key = key + torch.where((p & 8) != 0, 2, 0)
    columns = original.index_select(-1, key)
    mask = torch.arange(128, device=device)[None, :] < ranks.to(device)[:, None]
    columns = columns * mask[:, :, None]
    hi = columns.to(torch.bfloat16)
    lo = (columns - hi.float()).to(torch.bfloat16)
    packed_bf = packed[:, 16384:33792].view(torch.bfloat16).view(heads, 2, 128, 136)
    packed_bf[:, 0, :, :128].copy_(hi)
    packed_bf[:, 1, :, :128].copy_(lo)
    packed_piv = packed[:, 33792:].view(heads, 8, 136)
    packed_piv[:, :, :128].copy_(columns[:, :8, :])
    return packed


class KDASketchTables(torch.nn.Module):
    """Ranks, packed frames and launch tables of one KDA layer.

    Ranks ``0`` and ``128`` mean dense.
    """

    def __init__(
        self,
        frames: torch.Tensor,
        ranks: torch.Tensor,
        window: int = KDA_SKETCH_WINDOW,
    ):
        super().__init__()
        _check_window(window)
        self.window = window
        cpu = ranks.detach().to("cpu", torch.int64)
        if frames.shape[1:] != (128, 128) or cpu.shape != (frames.shape[0],):
            raise ValueError("KDA sketch needs (H, 128, 128) frames and H ranks")
        if ((cpu < 0) | (cpu > 128)).any():
            raise ValueError("KDA sketch ranks must be in [0, 128]")
        cpu = torch.where(cpu == 128, 0, cpu)
        self.num_heads = len(cpu)
        if self.num_heads > 128:
            raise ValueError("KDA sketch kernels support at most 128 heads")
        self.rank_cap = kda_rank_cap(cpu)
        device = frames.device
        # Flush buckets: CPU int32 (head | rank << 16) per head.
        self.flush_groups: list[tuple[torch.Tensor, int, int]] = []
        heads_all: list[int] = []
        for lo, hi in KDA_SKETCH_BUCKETS:
            heads = torch.where((cpu > lo) & (cpu <= hi))[0]
            if heads.numel():
                packed = (heads + (cpu[heads] << 16)).to(torch.int32).contiguous()
                self.flush_groups.append((packed, hi, len(heads_all)))
                heads_all += heads.tolist()
        all_ids = torch.tensor(heads_all, dtype=torch.int64, device="cpu")
        self.heads_all = (all_ids + (cpu[all_ids] << 16)).to(torch.int32)
        # Step: heads by rank, dense heads last.
        order = torch.argsort(
            torch.where(cpu > 0, cpu, torch.full_like(cpu, 1 << 20)), stable=True
        )
        self.register_buffer(
            "ranks", cpu.to(device=device, dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "heads_step", order.to(device=device, dtype=torch.int32), persistent=False
        )
        self.register_buffer("frame_gk", _pack_frames(frames, cpu), persistent=False)
        self.register_buffer("heads_all_d", self.heads_all.to(device), persistent=False)

    @property
    def num_sketch_heads(self) -> int:
        return self.heads_all.numel()

    def sketch_specs(self) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """Per-request ``name: (shape, dtype)`` of the sketch."""
        h, g, d, w = self.num_heads, self.rank_cap, KDA_SKETCH_HEAD_DIM, self.window
        return {
            "u": ((h, g, d), torch.bfloat16),
            "phi": ((h, g, d), torch.bfloat16),
            "f": ((h, g, w), torch.bfloat16),
        }


def kda_sketch_ring_specs(
    num_heads: int,
    window: int = KDA_SKETCH_WINDOW,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Per-state-slot ``name: (shape, dtype)`` of the window rings."""
    _check_window(window)
    h, w, d = num_heads, window, KDA_SKETCH_HEAD_DIM
    bf16, fp32 = torch.bfloat16, torch.float32
    return {
        "k": ((h, w, d), bf16),
        "v": ((h, w, d), bf16),
        "prefix": ((h, w, d), fp32),
        "beta": ((h, w), fp32),
        "u_ring": ((h, w, d), bf16),
        "d_ring": ((h, w, d), bf16),
    }


def kda_sketch_paged_views(
    specs: Sequence[tuple[tuple[int, ...], torch.dtype]],
    num_slots: int,
    page_bytes: int | None = None,
    device=None,
) -> list[torch.Tensor]:
    """Zeroed per-slot tensors packed into one page per slot."""
    sizes = [math.prod(shape) * dtype.itemsize for shape, dtype in specs]
    page = sum(sizes) if page_bytes is None else page_bytes
    assert page >= sum(sizes) and all(s % 16 == 0 for s in sizes) and page % 16 == 0
    raw = torch.zeros(num_slots * page, dtype=torch.uint8, device=device)
    views, offset = [], 0
    for (shape, dtype), size in zip(specs, sizes):
        item = dtype.itemsize
        contiguous = torch.empty(shape, device="meta").stride()
        views.append(
            torch.as_strided(
                raw.view(dtype),
                (num_slots, *shape),
                (page // item, *contiguous),
                offset // item,
            )
        )
        offset += size
    return views


@dataclass
class KDASketchRings:
    """One KDA layer's window rings (first dim: state slot).

    All rings must share the same slot stride in bytes.
    """

    k: torch.Tensor
    v: torch.Tensor
    prefix: torch.Tensor
    beta: torch.Tensor
    u_ring: torch.Tensor
    d_ring: torch.Tensor

    @property
    def window(self) -> int:
        return self.k.shape[2]

    def tensors(self) -> tuple[torch.Tensor, ...]:
        # Not dataclasses.astuple, which deep-copies the tensors.
        return tuple(getattr(self, f.name) for f in fields(self))


@dataclass
class KDASketchArgs:
    """One KDA layer's per-request sketch (first dim: request index)."""

    u: torch.Tensor
    phi: torch.Tensor
    f: torch.Tensor
    tables: KDASketchTables

    @classmethod
    def allocate(
        cls, tables: KDASketchTables, max_num_reqs: int, device=None
    ) -> "KDASketchArgs":
        buffers = {}
        for name, (shape, dtype) in tables.sketch_specs().items():
            numel = max_num_reqs * math.prod(shape)
            # The step stages 16 rows per chunk: 16 spare rows after u / phi.
            spare = 16 * KDA_SKETCH_HEAD_DIM if name in ("u", "phi") else 0
            flat = torch.zeros(numel + spare, dtype=dtype, device=device)
            buffers[name] = flat[:numel].view(max_num_reqs, *shape)
        return cls(**buffers, tables=tables)


def kda_sketch_scratch_numel(max_rows: int, num_sketch_heads: int) -> int:
    """FP32 scratch floats between a flush and its finish for ``max_rows``."""
    return max(1, max_rows * num_sketch_heads * KDA_SKETCH_SCRATCH_F)


def kda_sketch_scratch(
    max_rows: int, tables: KDASketchTables, device=None
) -> torch.Tensor:
    """FP32 flush scratch for up to ``max_rows`` rows, shared across layers."""
    n = kda_sketch_scratch_numel(max_rows, tables.num_sketch_heads)
    return torch.empty(n, dtype=torch.float32, device=device)


@triton.jit
def _fold_window_kernel(Slots, Pos, State, KR, VR, PrefixR, BR, Ranks,
                        H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                        W: tl.constexpr, BV: tl.constexpr, S0, S1,
                        S2: tl.constexpr, S3: tl.constexpr, RK, RV, RP,
                        RB):  # fmt: skip
    # Apply a sketch head's pending window rows ``t < pos`` to its state.
    row = tl.program_id(0)
    head = tl.program_id(1)
    block = tl.program_id(2)
    page = tl.load(Slots + row)
    if page > 0:
        count = tl.load(Pos + row)
        ready = count > 0
        if ready & (tl.load(Ranks + head) > 0):
            kk = tl.arange(0, K)
            vv = block * BV + tl.arange(0, BV)
            sp = (
                State
                + page.to(tl.int64) * S0
                + head * S1
                + vv[:, None] * S2
                + kk[None, :] * S3
            )
            state = tl.load(sp, mask=vv[:, None] < V, other=0.0)
            kr = KR + page.to(tl.int64) * RK + head * W * K
            vr = VR + page.to(tl.int64) * RV + head * W * V
            pr = PrefixR + page.to(tl.int64) * RP + head * W * K
            br = BR + page.to(tl.int64) * RB + head * W
            for t in range(count):
                k = tl.load(kr + t * K + kk).to(tl.float32)
                k *= tl.rsqrt(tl.sum(k * k) + 1e-6)
                v = tl.load(vr + t * V + vv, mask=vv < V, other=0.0).to(tl.float32)
                prefix = tl.load(pr + t * K + kk)
                prev = tl.load(pr + (t - 1) * K + kk, mask=t > 0, other=0.0)
                log_a = prefix - prev
                beta = tl.load(br + t)
                state *= tl.exp(log_a[None, :])
                delta = beta * (v - tl.sum(state * k[None, :], axis=1))
                state += delta[:, None] * k[None, :]
            tl.store(sp, state, mask=vv[:, None] < V)


def kda_sketch_fold_window_(
    state: torch.Tensor,
    rings: KDASketchRings,
    slots: torch.Tensor,
    pos: torch.Tensor,
    tables: KDASketchTables,
) -> None:
    """Apply each row's pending window rows ``t < pos[row]`` to its state.

    The windows are not reset; the caller restarts them with a cold build.
    """
    batch = slots.numel()
    if batch == 0:
        return
    h = tables.num_heads
    r = rings
    _fold_window_kernel[(batch, h, 4)](
        slots, pos, state, r.k, r.v, r.prefix, r.beta, tables.ranks, H=h, K=128,
        V=128, W=r.window, BV=32, S0=state.stride(0), S1=state.stride(1),
        S2=state.stride(2), S3=state.stride(3), RK=r.k.stride(0),
        RV=r.v.stride(0), RP=r.prefix.stride(0), RB=r.beta.stride(0),
        num_warps=4,
    )  # fmt: skip
