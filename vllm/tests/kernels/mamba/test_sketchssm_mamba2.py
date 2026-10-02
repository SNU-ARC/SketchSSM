# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM Mamba-2 decode (CUDA and Triton kernels) against FP64 references."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops import sketchssm_kernels as skk
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2 as sk
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_triton as skt
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")

H, P, G, N, NULL = 32, 32, 8, 128, 0
RANKS = ([0, 1, 2, 3, 4, 5, 9, 20, 38, 64] * H)[:H]  # dense, maps, pivots
DECODE = {"cuda": skk.mamba2_cuda_decode, "triton": skt.sketch_triton_decode}


def sketch_read(s: torch.Tensor, q: torch.Tensor, m: int) -> torch.Tensor:
    """Four-pivot residual-diagonal read of state ``s`` (P, N) with ``q``."""
    if m == 0:
        return s @ q
    norm = s / s.square().sum(0).mean().sqrt()
    dirs: list[torch.Tensor] = []
    for u in norm[:, : min(m, sk.SKETCH_PIVOTS)].T:
        v = u.clone()
        for d in dirs + dirs:
            v -= d * (d @ v)
        keep = v.square().sum() > 1e-12 * u.square().sum()
        dirs.append(v / v.norm() if keep else torch.zeros_like(v))
    z = torch.stack(dirs) @ norm
    residual = (norm.square().sum(0) - z.square().sum(0)).clamp_min(0)
    residual[: len(dirs)] = 0
    metric = torch.diag(residual) + z.T @ z
    ridge = 0.1 * torch.eye(m, dtype=s.dtype, device=s.device)
    return s[:, :m] @ torch.linalg.solve(metric[:m, :m] + ridge, metric[:m] @ q)


def stored_read(sketch: sk.SketchArgs, meta: int, h: int, q: torch.Tensor):
    """Read of head ``h`` with ``q`` from the stored sketch row ``meta``."""
    t, m = sketch.tables, RANKS[h]
    U = sketch.u[meta, int(t.u_offsets[h]) :].double()
    if m == 0:
        return U[:N].T @ q
    w_off, ag_off = int(t.w_offsets[h]), int(t.ag_offsets[h])
    coeff = sketch.w[meta, w_off : w_off + min(m, 4)].double() @ q
    if m > 4:
        ag = sketch.ag[meta, :, ag_off : ag_off + m].double()
        coeff = ag[0] * q[:m] + (ag[1:] * coeff[:, None]).sum(0)
    return U[:m].T @ coeff


class Layer:
    """One layer's state, rings and sketch; the last row pads the batch."""

    def __init__(self, backend: str, W: int, batch: int = 5):
        bf16, S = torch.bfloat16, batch + 1
        cuda_ok = skk.mamba2_cuda_supported(H, P, N, G, W, bf16, torch.float32)
        if backend == "cuda" and not cuda_ok:
            pytest.skip("Requires the sketchssm package")
        self.decode, self.W, self.batch = DECODE[backend], W, batch
        g = self.g = torch.Generator().manual_seed(0)
        rand = lambda *s: torch.rand(*s, generator=g).cuda()  # noqa: E731
        randn = lambda *s: torch.randn(*s, generator=g).cuda() * 0.1  # noqa: E731
        # Key-major FP32 state; kernels see the (slot, H, dim, dstate) view.
        self.state = randn(S, H, N, P).transpose(-1, -2)
        self.x_cache, self.dt_cache = randn(S, H, W, P).to(bf16), rand(S, H, W) * 0.1
        self.B_cache, self.A = randn(S, G, W, N).to(bf16), -rand(H) - 0.5
        self.dt_bias, self.D = (rand(H) * 0.1).to(bf16), rand(H).to(bf16)
        self.slots = torch.tensor([*range(1, batch), NULL], dtype=torch.int32).cuda()
        self.meta = torch.arange(batch, dtype=torch.int32).flip(0).cuda()
        ranks = torch.tensor(RANKS, dtype=torch.int32)
        shapes = zip(sk.sketch_shapes(ranks, P, N), sk.SKETCH_DTYPES)
        u, w, ag = (torch.zeros(batch, *s, dtype=d).cuda() for s, d in shapes)
        self.sketch = sk.SketchArgs(u, w, ag, sk.SketchTables(ranks, N).cuda())
        # x, B and C are slices of one row buffer, as in the model.
        self.conv = torch.empty(batch, H * P + 2 * G * N, dtype=bf16).cuda()
        self.out = torch.empty(batch, H, P, dtype=bf16).cuda()
        self.bc_pre = torch.empty(batch, G, W).cuda()

    def inputs(self):
        self.conv.copy_(torch.randn(self.conv.shape, generator=self.g) * 0.3)
        x, B, C = self.conv.split([H * P, G * N, G * N], 1)
        dt = (torch.randn(self.batch, H, generator=self.g) * 0.5).cuda().bfloat16()
        return x.view(-1, H, P), dt, B.view(-1, G, N), C.view(-1, G, N)

    def build(self, rows: torch.Tensor | None = None) -> None:
        flags = torch.ones(self.batch, dtype=torch.int8).cuda()
        sk.sketch_build(self.state, flags, self.slots, self.meta, self.sketch, NULL,
                        rows=rows)  # fmt: skip

    def step(self, x, dt, B, C, pos, flush):
        rows = sk.flush_row_list(flush)
        self.decode(self.state, x, dt[..., None].expand(-1, -1, P),
                    self.A[:, None, None].expand(-1, P, N), B, C,
                    self.D[:, None].expand(-1, P), self.dt_bias[:, None].expand(-1, P),
                    self.x_cache, self.dt_cache, self.B_cache, self.bc_pre, pos,
                    flush, rows, self.slots, self.meta, self.out, self.sketch,
                    NULL)  # fmt: skip
        return self.out

    def checked_step(self, pos: torch.Tensor) -> None:
        """Decode at ring positions ``pos`` and check against FP64 references."""
        x, dt, B, C = self.inputs()
        s0, flush = self.state.double(), (pos == self.W - 1).to(torch.int8).cuda()
        got = self.step(x, dt, B, C, pos.int().cuda(), flush).double()
        assert torch.equal(self.state[NULL].double(), s0[NULL])
        for row, slot in enumerate(self.slots.tolist()[:-1]):
            p, r = int(pos[row]), H // G
            dt_cur = F.softplus(dt[row].float() + self.dt_bias.float())[:, None]
            dts = torch.cat([self.dt_cache[slot, :, :p], dt_cur], 1).double()
            cs, a = dts.cumsum(1), self.A.double()[:, None]
            w, decay = dts * torch.exp(a * (cs[:, -1:] - cs)), torch.exp(a * cs[:, -1:])
            xs = torch.cat([self.x_cache[slot, :, :p], x[row, :, None]], 1).double()
            ks = torch.cat([self.B_cache[slot, :, :p], B[row, :, None]], 1).double()
            ks, q = ks.repeat_interleave(r, 0), C[row].double().repeat_interleave(r, 0)
            skip = self.D.double()[:, None] * xs[:, -1]
            if p == self.W - 1:
                s = decay[..., None] * s0[slot]
                s += torch.einsum("htp,ht,htn->hpn", xs, w, ks)
                got_s = self.state[slot].double()
                torch.testing.assert_close(got_s, s, rtol=1e-5, atol=1e-6)
                want, tol = torch.einsum("hpn,hn->hp", s, q) + skip, [1e-2] * H
            else:
                ring = torch.einsum("htp,ht,htn,hn->hp", xs, w, ks, q)
                # The kernel reads its stored sketch (BF16 output rounding) ...
                meta = int(self.meta[row])
                read = [stored_read(self.sketch, meta, h, q[h]) for h in range(H)]
                want = decay * torch.stack(read) + ring + skip
                err = (got[row] - want).abs().amax(1) / want.abs().amax(1)
                assert (err < 5e-3).all(), (row, p, err)
                # ... which holds the four-pivot read up to BF16 storage.
                read = [sketch_read(s0[slot, h], q[h], m) for h, m in enumerate(RANKS)]
                want = decay * torch.stack(read) + ring + skip
                tol = [2e-2 if m == 0 else 5e-2 for m in RANKS]
                assert torch.equal(self.state[slot].double(), s0[slot])
            err = (got[row] - want).abs().amax(1) / want.abs().amax(1)
            assert (err.cpu() < torch.tensor(tol)).all(), (row, p, err)


@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_sketchssm_mamba2_decode(backend):
    """Rows at mixed window positions decode a full window, each flushing."""
    W = 16
    layer = Layer(backend, W)
    layer.build()
    for t in range(W + 1):
        layer.checked_step((t + torch.tensor([0, W // 2 + 1, W - 1, 3, 0])) % W)


@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_sketchssm_mamba2_decode_after_prefill_build(backend):
    """The sketch built from a prefilled state serves the next decode."""
    layer = Layer(backend, 16)
    layer.build(rows=torch.arange(5, dtype=torch.int32).cuda())
    assert not layer.sketch.u[int(layer.meta[-1])].any()  # padding row
    layer.checked_step(torch.tensor([0, 5, 11, 14, 0]))
