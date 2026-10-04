# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""SketchSSM Mamba-2 CUDA decode against FP64 oracles.

Rows sit at different window positions, so every step mixes flush and
non-flush rows; windows longer than 16 replay the ring in several 16-step
tiles. A non-flush read must match the read of the stored sketch
(U, maps, AG) up to BF16 output rounding, and the stored sketch must match
the four-pivot residual-diagonal read solved directly (``sketch_coeff``) up
to its BF16 storage; dense heads read the plain state. Flushes are checked
against the exact FP64 replay of the window. The state lives in random
per-group frames R; B and C arrive unrotated, the oracles rotate in FP64.
"""

import json

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("vllm")

from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2 as sk

from sketchssm import kernels
from sketchssm.kernels import mamba2 as mk

RIDGE = 0.1

N, L = 128, 16
NULL = 0
# (heads, head dim, groups): Nemotron 3 Super, Nemotron Nano 9B v2, and a
# smaller layer with 32-dim heads in groups of four.
SHAPES = [(128, 64, 8), (128, 80, 8), (32, 32, 8)]
# (shape, state size) of layers with other state sizes: Falcon-H1-like
# 128-dim heads in one or two groups, and a Zamba2-like layer.
STATE_SHAPES = [((24, 128, 1), 256), ((32, 128, 2), 256), ((64, 64, 1), 64)]
LONG_WINDOWS = [32, 64]
SLOW = pytest.mark.slow


def window_params(window: int, *args):
    return pytest.param(*args, window, marks=SLOW if window > L else ())


# (shape, state size, window)
CASES = [
    window_params(w, s, n)
    for w in [L, *LONG_WINDOWS]
    for s, n in [(s, N) for s in SHAPES] + STATE_SHAPES
]


def skip_unsupported(shape, state_size, window) -> None:
    H, P, G = shape
    support = kernels.mamba2_supported(
        H, P, state_size, G, window, torch.bfloat16, torch.float32
    )
    if not support:
        pytest.skip(support.reason)


def decode(state, x, dt, A, B, C, D, dt_bias, x_cache, dt_cache, B_cache, bc_pre,
           write_pos, is_flush, flush_rows, slots, meta, out, sketch):  # fmt: skip
    """vLLM's call: fill ``bc_pre`` and the FP32 query, then the decode with
    its flush stream."""
    sk.sketch_bc_pre(B, C, B_cache, write_pos, is_flush, bc_pre, slots, NULL)
    query = sk.sketch_query(C, sketch.frames_t)
    kernels.mamba2_decode(
        state, x, dt, A, B, query, D, dt_bias, x_cache, dt_cache, B_cache, bc_pre,
        write_pos, is_flush, flush_rows, slots, meta, out, sketch, NULL,
        frames_t=sketch.frames_t, flush_programs=sk.row_list_programs(x.shape[0]),
        run_with_flush=sk.run_with_flush,
    )  # fmt: skip


def ranks_for(heads: int) -> list[int]:
    return ([0, 1, 2, 3, 4, 5, 9, 20, 38, 64] * heads)[:heads]


def sketch_coeff(s: torch.Tensor, q: torch.Tensor, m: int) -> torch.Tensor:
    """Four-pivot residual-diagonal read coefficients of state ``s`` (P, N)."""
    mean = s.square().sum(0).mean()
    norm = s / torch.sqrt(mean) if mean > 0 else s
    directions: list[torch.Tensor] = []
    for j in range(min(m, sk.SKETCH_PIVOTS)):
        u = norm[:, j]
        v = u.clone()
        for _ in range(2):
            for d in directions:
                v -= d * (d @ v)
        keep = v.square().sum() > 1e-12 * u.square().sum()
        directions.append(v / v.norm() if keep else torch.zeros_like(v))
    z = torch.stack(directions) @ norm
    residual = (norm.square().sum(0) - z.square().sum(0)).clamp_min(0)
    residual[: min(m, sk.SKETCH_PIVOTS)] = 0
    metric = torch.diag(residual) + z.T @ z
    ridge = RIDGE * torch.eye(m, dtype=s.dtype)
    return torch.linalg.solve(metric[:m, :m] + ridge, metric[:m] @ q)


class Layer:
    def __init__(
        self,
        batch: int,
        shape: tuple[int, int, int],
        N: int = N,
        L: int = L,
        dtype: torch.dtype = torch.bfloat16,
    ):
        H, P, G = shape
        self.H, self.P, self.G = H, P, G
        self.N, self.L, self.dtype = N, L, dtype
        self.ranks = ranks_for(H)
        g = torch.Generator().manual_seed(0)
        self.g = g
        num_slots = batch + 3

        def randn(*shape, std=1.0):
            return (torch.randn(*shape, generator=g) * std).cuda()

        # Key-major FP32 state; kernels see the (slot, H, dim, dstate) view.
        self.state = randn(num_slots, H, N, P, std=0.1).transpose(-1, -2)
        self.x_cache = randn(num_slots, H, L, P, std=0.1).to(dtype)
        self.dt_cache = torch.rand(num_slots, H, L, generator=g).cuda() * 0.1
        self.B_cache = randn(num_slots, G, L, N, std=0.1).to(dtype)
        self.A = -torch.rand(H, generator=g).cuda() - 0.5
        self.dt_bias = torch.rand(H, generator=g).cuda().mul(0.1).to(dtype).float()
        self.D = torch.rand(H, generator=g).cuda().to(dtype).float()
        self.slots = torch.arange(1, batch + 1, dtype=torch.int32).cuda()
        self.batch = batch
        # Sketch rows are persistent request indices, unrelated to the slots.
        self.meta = torch.arange(batch, dtype=torch.int32).flip(0).cuda()
        ranks = torch.tensor(self.ranks, dtype=torch.int32)
        shapes = sk.sketch_shapes(ranks, P, N)
        u, w, ag = (
            torch.zeros(batch, *s, dtype=d).cuda()
            for s, d in zip(shapes, sk.SKETCH_DTYPES)
        )
        frames = torch.linalg.qr(torch.randn(G, N, N, generator=g))[0]
        frames_t = frames.transpose(-1, -2).contiguous().cuda()
        # Row vectors rotate as v @ frames_t.
        self.rot = frames_t.double().cpu()
        self.sketch = sk.SketchArgs(
            u, w, ag, tables=sk.SketchTables(ranks, N).cuda(), frames_t=frames_t
        )
        # B/C slices of one row buffer, as in the model.
        self.conv = torch.empty(batch, H * P + 2 * G * N).cuda().to(dtype)

    def inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        H, P, G, N = self.H, self.P, self.G, self.N
        conv = self.conv
        conv.copy_(torch.randn(conv.shape, generator=self.g) * 0.3)
        x = conv[:, : H * P].view(-1, H, P)
        B = conv[:, H * P : H * P + G * N].view(-1, G, N)
        C = conv[:, H * P + G * N :].view(-1, G, N)
        dt = (torch.randn(self.batch, H, generator=self.g) * 0.5).cuda().to(self.dtype)
        return x, dt, B, C

    def build(self, flags):
        sk.sketch_build(self.state, flags, self.slots, self.meta, self.sketch, NULL)

    def step(self, x, dt, B, C, write_pos, is_flush, out=None, bc_pre=None,
             flush_rows=None, slots=None, meta=None):  # fmt: skip
        P, G, N, L = self.P, self.G, self.N, self.L
        if out is None:
            out = torch.empty(x.shape, dtype=self.dtype).cuda()
        if bc_pre is None:
            bc_pre = torch.empty(self.batch, G, L).cuda()
        if flush_rows is None:
            flush_rows = sk.flush_row_list(is_flush)
        decode(
            self.state,
            x,
            dt[:, :, None].expand(-1, -1, P),
            self.A[:, None, None].expand(-1, P, N),
            B,
            C,
            self.D.to(self.dtype)[:, None].expand(-1, P),
            self.dt_bias.to(self.dtype)[:, None].expand(-1, P),
            self.x_cache,
            self.dt_cache,
            self.B_cache,
            bc_pre,
            write_pos,
            is_flush,
            flush_rows,
            self.slots if slots is None else slots,
            self.meta if meta is None else meta,
            out,
            self.sketch,
        )
        return out

    def stored_read(self, row, h, q):
        """Readout of head ``h`` from the stored sketch of row ``row``."""
        sketch, t, N = self.sketch, self.sketch.tables, self.N
        meta, m = int(self.meta[row]), self.ranks[h]
        u_off = int(t.u_offsets[h])
        U = sketch.u[meta].double().cpu()
        if m == 0:
            return U[u_off : u_off + N].T @ q
        w_off, ag_off = int(t.w_offsets[h]), int(t.ag_offsets[h])
        dots = sketch.w[meta, w_off : w_off + min(m, 4)].double().cpu() @ q
        coeff = dots
        if m > 4:
            ag = sketch.ag[meta, :, ag_off : ag_off + m].double().cpu()
            coeff = ag[0] * q[:m] + (ag[1:] * dots[:, None]).sum(0)
        return U[u_off : u_off + m].T @ coeff

    def window(self, slot, row, h, pos, x, dt, B):
        g = h // (self.H // self.G)
        dt_all = torch.cat(
            [
                self.dt_cache[slot, h, :pos].cpu(),
                F.softplus(dt[row, h].float() + self.dt_bias[h]).cpu()[None],
            ]
        ).double()
        cs = dt_all.cumsum(0)
        a = self.A[h].double().cpu()
        weights = dt_all * torch.exp(a * (cs[-1] - cs))
        values = torch.cat([self.x_cache[slot, h, :pos], x[row, h, None]])
        keys = torch.cat([self.B_cache[slot, g, :pos], B[row, g, None]])
        return (
            torch.exp(a * cs[-1]),
            weights,
            values.double().cpu(),
            keys.double().cpu() @ self.rot[g],
        )

    def query(self, C, row, h):
        """The FP64 rotated query of head ``h``."""
        g = h // (self.H // self.G)
        return C[row, g].double().cpu() @ self.rot[g]


@pytest.mark.parametrize(("shape", "state_size", "window"), CASES)
def test_mamba2_decode(shape, state_size, window):
    skip_unsupported(shape, state_size, window)
    batch = 5
    layer = Layer(batch, shape, state_size, window)
    L = window
    if L <= 16:
        offsets = torch.tensor([0, 5, 11, 15, 3])
    else:
        # Every row but one flushes within the 20 steps, and rows cross the
        # first tile boundary and read in the middle of the window.
        offsets = torch.tensor([L - 17, L // 2 + 1, L - 9, L - 1, 3])
    layer.build(torch.ones(batch, dtype=torch.int8).cuda())
    for t in range(20):
        if t == 9:
            # Rows are reordered; sketches follow the requests.
            perm = torch.tensor([3, 0, 4, 1, 2])
            layer.slots = layer.slots[perm.cuda()].contiguous()
            layer.meta = layer.meta[perm.cuda()].contiguous()
            offsets = offsets[perm]
        pos = (t + offsets) % L
        write_pos = pos.to(torch.int32).cuda()
        is_flush = (pos == L - 1).to(torch.int8).cuda()
        x, dt, B, C = layer.inputs()
        s0 = layer.state.double().cpu()
        stored = {
            (row, h): layer.stored_read(row, h, layer.query(C, row, h))
            for row in range(batch)
            for h in range(layer.H)
            if int(pos[row]) != L - 1
        }
        got = layer.step(x, dt, B, C, write_pos, is_flush).double().cpu()
        for row, slot in enumerate(layer.slots.tolist()):
            p = int(pos[row])
            for h, m in enumerate(layer.ranks):
                decay, w, values, keys = layer.window(slot, row, h, p, x, dt, B)
                q = layer.query(C, row, h)
                skip = layer.D[h].double().cpu() * values[-1]
                if p == L - 1:
                    s = decay * s0[slot, h] + (values * w[:, None]).T @ keys
                    torch.testing.assert_close(
                        layer.state[slot, h].double().cpu(), s, rtol=1e-5, atol=1e-6
                    )
                    want = s @ q + skip
                    err = (got[row, h] - want).abs().max() / want.abs().max()
                    assert err < 1e-2, (t, row, h, m, p, float(err))
                    continue
                ring = (values * (w * (keys @ q))[:, None]).sum(0)
                # The kernel reads its stored sketch (BF16 output rounding) ...
                want = decay * stored[row, h] + ring + skip
                err = (got[row, h] - want).abs().max() / want.abs().max()
                assert err < 5e-3, (t, row, h, m, p, float(err))
                # ... which holds the four-pivot read up to BF16 storage.
                s = s0[slot, h]
                read = s @ q if m == 0 else s[:, :m] @ sketch_coeff(s, q, m)
                want = decay * read + ring + skip
                err = (got[row, h] - want).abs().max() / want.abs().max()
                assert err < (2e-2 if m == 0 else 5e-2), (t, row, h, m, p, float(err))
            if p != L - 1:
                assert torch.equal(layer.state[slot], s0[slot].float().cuda())


@pytest.mark.parametrize(("shape", "state_size", "window"), CASES)
def test_mamba2_build_matches_flush(shape, state_size, window):
    """The cold build (after a prefill) and the fused flush write the same
    sketch for the same state."""
    skip_unsupported(shape, state_size, window)
    batch = 3
    layer = Layer(batch, shape, state_size, window)
    pos = torch.full((batch,), window - 1, dtype=torch.int32).cuda()
    flags = torch.ones(batch, dtype=torch.int8).cuda()
    layer.step(*layer.inputs(), pos, flags)
    sketch = layer.sketch
    flushed = [t.clone() for t in (sketch.u, sketch.w, sketch.ag)]
    for t in (sketch.u, sketch.w, sketch.ag):
        t.zero_()
    layer.build(flags)
    # Over longer windows the pivot factors (AG) of some rank-64 heads are
    # ill-conditioned: the two FP32 summation orders differ by more than 1e-4
    # there, while the reads they give agree to 5e-5.
    atols = (1e-4, 1e-4, 1e-4 if window == L else 1e-3)
    for got, want, atol in zip((sketch.u, sketch.w, sketch.ag), flushed, atols):
        torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=atol)


def test_mamba2_window16_config_fallback(config_dir, clear_caches, monkeypatch):
    """A window without its own tuned file takes the window-16 file of the
    layer shape; knobs from it that do not fit the longer window (non-flush
    shared memory past 48 KB, flush shared memory past the GPU's limit) give
    way to the architecture defaults."""
    shape, state_size, window = (32, 128, 2), 256, 64
    H, P, G = shape
    skip_unsupported(shape, state_size, window)
    name = mk.config_file_name(P, H // G, state_size)
    knobs = {"nf": {"NF_HEADS": 16}, "flush": {"WARPS": 4}}
    (config_dir / name).write_text(json.dumps(knobs))
    built: list[tuple[str, int, int]] = []
    build = mk._build

    def record(name, config, fns):
        ext = build(name, config, fns)
        built.append((name, config.get("NF_HEADS", 0), config.get("WARPS", 0)))
        return ext

    monkeypatch.setattr(mk, "_build", record)
    clear_caches(mk.tuned_config, mk.large_batch_config, mk.window16_fallback, mk._nf_ext,
                 mk._flush_ext)
    assert mk.window16_fallback(P, H // G, state_size, window)
    nf, flush = mk.tuned_config(P, H // G, state_size, window)
    assert (nf["NF_HEADS"], flush["WARPS"]) == (16, 4)
    layer = Layer(3, shape, state_size, window)
    pos = torch.tensor([3, window - 1, 40], dtype=torch.int32).cuda()
    out = layer.step(*layer.inputs(), pos, (pos == window - 1).to(torch.int8))
    assert torch.isfinite(out).all()
    defaults = mk.default_configs()
    # The non-flush build with 16 heads fails; the flush build with four warps
    # builds but does not fit, so both run with the defaults.
    assert built == [
        ("mamba2_sketch_nf", defaults[0]["NF_HEADS"], 0),
        ("mamba2_sketch_flush", 0, 4),
        ("mamba2_sketch_flush", 0, defaults[1]["WARPS"]),
    ]


@pytest.mark.parametrize(
    "window", [L, *(pytest.param(w, marks=SLOW) for w in LONG_WINDOWS)]
)
@torch.inference_mode()
def test_mamba2_cuda_graph_lifecycle(window):
    """One decode captured in a CUDA graph and replayed for more than a window,
    with rows at mixed positions, reordered rows, a padding row and a changing
    flush list, matches eager decode bitwise: outputs, state, rings and
    sketch."""
    batch, shape = 6, (32, 32, 8)
    skip_unsupported(shape, N, window)
    layer = Layer(batch, shape, N, window)
    eager = Layer(batch, shape, N, window)
    flags = torch.ones(batch, dtype=torch.int8).cuda()
    layer.build(flags)
    eager.build(flags)
    x, dt, B, C = layer.inputs()
    dt = dt.clone()
    pos_buf = torch.zeros(batch, dtype=torch.int32).cuda()
    flush_buf = torch.zeros(batch, dtype=torch.int8).cuda()
    rows_buf = torch.full((batch,), -1, dtype=torch.int32).cuda()
    slots_buf, meta_buf = layer.slots.clone(), layer.meta.clone()
    out = torch.empty(x.shape, dtype=layer.dtype).cuda()
    bc_pre = torch.empty(batch, layer.G, window).cuda()

    def launch():
        layer.step(x, dt, B, C, pos_buf, flush_buf, out, bc_pre, rows_buf,
                   slots_buf, meta_buf)  # fmt: skip

    def tensors(lay):
        sketch = lay.sketch
        return (lay.state, lay.x_cache, lay.dt_cache, lay.B_cache, sketch.u,
                sketch.w, sketch.ag)  # fmt: skip

    saved = [t.clone() for t in tensors(layer)]
    launch()  # compile before capture, then restore
    for t, v in zip(tensors(layer), saved):
        t.copy_(v)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for t, v in zip(tensors(layer), saved):
        t.copy_(v)
    gen = torch.Generator().manual_seed(1)
    offsets = torch.randint(0, window, (batch,), generator=gen)
    offsets[:2] = torch.tensor([window - 1, window - 2])
    order = torch.arange(batch)
    for t in range(window + 8):
        if t % 5 == 0:
            order = torch.randperm(batch, generator=gen)
        pos = (t + offsets[order]) % window
        slots = (order + 1).to(torch.int32)
        meta = order.flip(0).to(torch.int32)
        pad = t % 3 == 0
        if pad:
            # The last row pads the captured batch (null slot, no flush).
            slots[-1], pos[-1] = NULL, 0
        is_flush = (pos == window - 1).to(torch.int8)
        if pad:
            is_flush[-1] = 0
        x_t, dt_t, B_t, C_t = eager.inputs()
        for buf, value in zip(
            (x, dt, B, C, pos_buf, flush_buf, slots_buf, meta_buf),
            (x_t, dt_t, B_t, C_t, pos, is_flush, slots, meta),
        ):
            buf.copy_(value)
        rows_buf.copy_(sk.flush_row_list(flush_buf))
        graph.replay()
        want = eager.step(x_t, dt_t, B_t, C_t, pos_buf.clone(), flush_buf.clone(),
                          slots=slots_buf.clone(), meta=meta_buf.clone())  # fmt: skip
        live = slots_buf.cpu() != NULL
        torch.testing.assert_close(out[live], want[live], rtol=0, atol=0)
        for got_t, want_t in zip(tensors(layer), tensors(eager)):
            torch.testing.assert_close(got_t, want_t, rtol=0, atol=0)
