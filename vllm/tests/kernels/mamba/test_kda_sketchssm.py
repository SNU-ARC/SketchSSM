# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA SketchSSM decode (CUDA and Triton kernels) against an FP64 oracle."""

import math
from collections.abc import Callable

import pytest
import torch

from vllm.model_executor.layers.mamba.ops import kda_sketchssm_common as common
from vllm.model_executor.layers.mamba.ops import kda_sketchssm_triton as kdt
from vllm.model_executor.layers.mamba.ops import sketchssm_kernels as skk
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="GPU")
D = 128
RANKS = [0, 3, 17, 40, 70, 128]  # dense heads and one per flush rank bucket
H = len(RANKS)
CUDA = skk.kda_cuda_supported(H, D, D, 16, torch.bfloat16, torch.float32)
NO_CUDA = pytest.mark.skipif(not CUDA, reason="Requires the sketchssm package")
BACKENDS = [pytest.param("cuda", marks=NO_CUDA), "triton"]
KERNELS: dict[str, tuple[Callable[..., None], Callable[..., None]]] = {
    "cuda": (skk.kda_cuda_cold_build, skk.kda_cuda_decode),
    "triton": (kdt.kda_sketch_triton_cold_build, kdt.kda_sketch_triton_decode),
}
i32 = lambda x: torch.tensor(x, device="cuda", dtype=torch.int32)
rel = lambda a, b: float((a.double().cpu() - b).norm() / b.norm().clamp_min(1e-30))


def coefficient(state, frame, rank, pivots=4):
    """FP64 coefficient map (K x rank) of a window-start state."""
    s = state @ frame
    mu = s.square().sum() / D
    s = s / torch.sqrt(mu if mu > 0 else torch.ones_like(mu))
    q: list[torch.Tensor] = []
    for i in range(min(rank, pivots)):
        v = s[:, i].clone()
        for b in q + q:
            v = v - b * (b @ v)
        keep = v.square().sum() > 1e-12 * s[:, i].square().sum()
        q.append(v / v.norm() if keep else torch.zeros_like(v))
    z = torch.stack(q) @ s
    res = (s.square().sum(0) - z.square().sum(0)).clamp_min(0)
    res[: min(rank, pivots)] = 0
    m = z.T @ z + torch.diag(res)
    eye = 0.1 * torch.eye(rank, dtype=torch.float64)
    return frame @ torch.linalg.solve(m[:rank, :rank] + eye, m[:rank]).T


class Sim:
    """Kernel buffers (slot 0 is the null block) next to an FP64 oracle."""

    def __init__(self, backend, window, pages, seed):
        torch.manual_seed(seed)
        frame = torch.linalg.qr(torch.randn(H, D, D, dtype=torch.float64)).Q
        initial = torch.randn(pages, H, D, D) * 0.1
        self.a, self.bias = torch.randn(H) * 0.2, torch.randn(H * D) * 0.1
        self.build, self.decode = KERNELS[backend]
        rings = common.kda_sketch_ring_specs(H, window)
        specs = [((H, D, D), torch.float32), *rings.values()]
        page = (sum(math.prod(s) * t.itemsize for s, t in specs) + 511) // 512 * 512
        views = common.kda_sketch_paged_views(specs, pages, page + 1024, "cuda")
        self.state, self.rings = views[0].copy_(initial), common.KDASketchRings(*views[1:])
        frames = frame.mT.float().contiguous().cuda()
        self.tables = common.KDASketchTables(frames, torch.tensor(RANKS), window)
        self.sketch = common.KDASketchArgs.allocate(self.tables, 8, "cuda")
        self.scratch = common.kda_sketch_scratch(8, self.tables, "cuda")
        self.frame, self.ref = frame.float().double(), initial.double()
        self.est = self.ref.clone()
        self.w, self.req, self.pos, self.maps = window, {}, {}, {}

    def admit(self, pages, reqs):
        """Cold build at window start, with -1 padding in the row list."""
        rows = i32(list(range(len(pages))) + [-1] * 3)
        scratch = common.kda_sketch_scratch(rows.numel(), self.tables, "cuda")
        slots, meta = i32(pages + [0] * 3), i32(reqs + [0] * 3)
        self.build(self.state, self.rings, slots, meta, rows, self.sketch, scratch)
        self.req.update(zip(pages, reqs))
        for p in pages:
            self.restart(p)

    def restart(self, p):
        s, f, self.pos[p], self.maps[p] = self.ref[p], self.frame, 0, {}
        self.est[p] = s
        for h, m in ((h, m) for h, m in enumerate(RANKS) if m % D):
            c = self.maps[p][h] = coefficient(s[h], f[h], m)
            self.est[p, h] = s[h] @ f[h, :, :m] @ c.T

    def metadata(self, rows):
        """Slots, request rows, window positions and flush rows of a step."""
        pos = [self.pos.get(p, 0) for p in rows]
        flush = [i for i, p in enumerate(rows) if p > 0 and pos[i] == self.w - 1]
        flush += [-1] * (len(rows) - len(flush))
        return [i32(x) for x in (rows, [self.req.get(p, 0) for p in rows], pos, flush)]

    def kernel_step(self, rows, data):
        out, a, bias = torch.empty_like(data[2]), self.a.cuda(), self.bias.cuda()
        meta = (*self.metadata(rows), self.sketch, self.scratch)
        self.decode(*data, a, bias, out, self.state, self.rings, *meta)
        return out

    def oracle_step(self, rows, data):
        q, k, v, g, beta = (x.double().cpu() for x in data)
        q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) / D**0.5
        k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        g = self.a.double().exp()[:, None] * (g + self.bias.double().view(H, D))
        decay, beta, out = torch.exp(-5 * torch.sigmoid(g)), beta.sigmoid(), 0 * v
        for b, p in ((b, p) for b, p in enumerate(rows) if p > 0):
            for x in (self.ref, self.est):
                y = x[p] * decay[b][:, None]
                err = v[b] - (y @ k[b, :, :, None])[..., 0]
                x[p] = y + (beta[b, :, None] * err)[..., None] * k[b, :, None]
            last, self.pos[p] = self.pos[p] == self.w - 1, self.pos[p] + 1
            out[b] = ((self.ref if last else self.est)[p] @ q[b, :, :, None])[..., 0]
            if last:
                self.restart(p)
        return out

    def check_maps(self, p):
        for h, want in self.maps[p].items():
            assert rel(self.sketch.phi[self.req[p], h, : want.shape[1]].T, want) < 4e-3

    def check_step(self, rows):
        """Check outputs, and the state and maps of the rows that flushed."""
        data = inputs(len(rows))
        out, want = self.kernel_step(rows, data), self.oracle_step(rows, data)
        for r, p in enumerate(rows):
            assert rel(out[r], want[r]) < 0.007 if p > 0 else not out[r].any(), (r, p)
            if p > 0 and self.pos[p] == 0:
                got = self.state[p].double().cpu()
                torch.testing.assert_close(got, self.ref[p], rtol=1e-3, atol=4e-6)
                self.check_maps(p)


def inputs(batch):
    q, k, v, g = (torch.randn(batch, H, D, device="cuda").bfloat16() for _ in range(4))
    return q, k, v, g - 3.0, torch.randn(batch, H, device="cuda").bfloat16()


@pytest.mark.parametrize("window", [16, 32])
@pytest.mark.parametrize("backend", BACKENDS)
def test_kda_sketch_decode(backend, window):
    """Mixed window positions, reordered and padded rows, through a flush."""
    sim = Sim(backend, window, pages=6, seed=7)
    live: list = []
    for t in range(7 * window // 4):
        if t < window and t % (window // 4) == 0:
            new = [1, 2] if t == 0 else [len(live) + 1]
            sim.admit(new, [7 - p for p in new])
            live += new
        rows = [live[i] for i in torch.randperm(len(live)).tolist()]
        rows.insert(t % (len(rows) + 1), 0)
        sim.check_step(rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_kda_sketch_cold_build_after_prefill(backend):
    """A re-admitted request is rebuilt next to a request mid-window."""
    sim = Sim(backend, 16, pages=3, seed=23)
    sim.admit([1, 2], [5, 3])
    for _ in range(5):
        sim.check_step([2, 1])
    prefilled = torch.randn(H, D, D) * 0.1
    prefilled[1] = 0
    sim.state[2], sim.ref[2] = prefilled, prefilled
    sim.admit([2], [3])
    sim.check_maps(2)
    for _ in range(17):
        sim.check_step([1, 0, 2])


@pytest.mark.parametrize("backend", BACKENDS)
def test_kda_dense_heads_change_state_only_at_flush(backend):
    """Dense heads read their BF16 state rows and keep the FP32 state fixed
    between flushes; the flush rebuilds the BF16 rows from the folded state."""
    sim = Sim(backend, 16, pages=4, seed=31)
    sim.admit([1, 2, 3], [4, 2, 6])
    dense = sim.tables.dense_heads.tolist()
    assert dense == [0, 5]

    def check_rows(pages):
        for p in pages:
            rows = sim.sketch.dense[sim.req[p]]
            assert torch.equal(rows, sim.state[p, dense].bfloat16())

    check_rows([1, 2, 3])
    for t in range(40):
        rows = [1, 2, 3] if t < 3 else [3, 1, 0, 2]
        before = sim.state[:, dense].clone()
        sim.check_step(rows)
        for p in rows:
            if p > 0 and sim.pos[p] != 0:
                assert torch.equal(sim.state[p, dense], before[p])
            elif p > 0:
                check_rows([p])
