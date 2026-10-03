# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN SketchSSM decode (CUDA and Triton) against an FP64 oracle."""

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops import gdn_sketchssm_common as common
from vllm.model_executor.layers.mamba.ops import sketchssm_kernels as skk
from vllm.model_executor.layers.mamba.ops.gdn_sketchssm_triton import (
    gdn_sketch_triton_decode,
)
from vllm.platforms import current_platform

CUDA_OK = skk.gdn_cuda_supported(2, 6, 128, 128, 16, torch.bfloat16, torch.float32)
pytestmark = pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="GPU")
NO_CUDA = pytest.mark.skipif(not CUDA_OK, reason="Requires the sketchssm package")
BACKENDS = [pytest.param("cuda", marks=NO_CUDA), "triton"]
DECODE: dict[str, Callable[..., None]] = {
    "cuda": skk.gdn_cuda_decode,
    "triton": gdn_sketch_triton_decode,
}
K = V = 128
NX, NR = 8, 5  # state slots (0 is the null block), sketch rows
# Ranks of 3 value heads per key head: dense, merged and four-pivot maps, full.
WIDTHS = [0, 3, 20, 128, 7, 1]
c = lambda x: x.double().cpu()
bf = lambda x: x.bfloat16().double()
unit = lambda x: x / torch.sqrt(x.square().sum(-1, keepdim=True) + 1e-6)
relative = lambda a, b: float((a - b).norm() / b.norm().clamp_min(1e-12))
i32 = lambda x: torch.tensor(x, device="cuda", dtype=torch.int32)
Request = lambda slot, meta, start=0: SimpleNamespace(slot=slot, meta=meta, start=start)
PAD = Request(0, 0)


def coefficient(s: torch.Tensor, m: int) -> torch.Tensor:
    """FP64 four-pivot residual-diagonal coefficient map (ridge 0.1)."""
    mean = s.square().sum(0).mean()
    s = s / torch.sqrt(mean if mean > 0 else torch.ones_like(mean))
    ds: list[torch.Tensor] = []
    for j in range(min(m, 4)):
        v = s[:, j].clone()
        for d in ds + ds:
            v -= d * (d @ v)
        keep = v.square().sum() > 1e-12 * s[:, j].square().sum()
        ds.append(v / v.norm() if keep else torch.zeros_like(v))
    z = torch.stack(ds) @ s
    res = (s.square().sum(0) - z.square().sum(0)).clamp_min(0)
    res[: min(m, 4)] = 0
    gram = z.T @ z + torch.diag(res)
    return torch.linalg.solve(gram[:m, :m] + 0.1 * torch.eye(m).double(), gram[:m])


class Harness:
    """One GDN layer (3 value heads per key head) and its oracle bookkeeping."""

    def __init__(self, backend, window, dtype=torch.bfloat16, widths=WIDTHS,
                 degenerate=False):  # fmt: skip
        self.decode_fn = DECODE[backend]
        self.dtype, self.widths, self.W = dtype, widths, window
        self.HV, self.H = len(widths), len(widths) // 3
        torch.manual_seed(3)
        self.state = torch.randn(NX, self.HV, V, K, device="cuda") * 0.12
        if degenerate:  # zero and repeated pivot columns, an all-zero head
            st = self.state
            st[..., 0], st[..., 2], st[:, 2] = 0, st[..., 1], 0
        with torch.device("cuda"):
            self.tables = common.GDNSketchTables(torch.tensor(widths), self.H, window)
        self.sketch = common.GDNSketchArgs.allocate(self.tables, NR, "cuda")
        self.dr = torch.full((NX, self.HV, window, V), torch.nan).bfloat16().cuda()
        self.kr = torch.full((NX, self.H, window, K), torch.nan).bfloat16().cuda()
        self.gr = torch.zeros(NX, self.HV, window, device="cuda")
        self.al = torch.randn(self.HV, device="cuda") * 0.05
        self.bias = torch.zeros(self.HV, device="cuda")

    def build(self, reqs, flags: list[int]) -> None:
        slots, meta = i32([r.slot for r in reqs]), i32([r.meta for r in reqs])
        common.gdn_sketch_build(self.state, i32(flags), slots, meta, self.sketch)

    def inputs(self, n: int) -> list[torch.Tensor]:
        mix = torch.randn(n, (2 * self.H + self.HV) * K, device="cuda") * 0.25
        a = torch.randn(n, self.HV, device="cuda") * 0.2 - 3
        b = torch.randn(n, self.HV, device="cuda") * 0.5
        return [x.to(self.dtype) for x in (mix, a, b)]

    def decode(self, mix, a, b, out, rows, positions):
        """The decode step as a closure over fixed index tensors."""
        flush = [i for i, (r, t) in enumerate(zip(rows, positions))
                 if r.slot and t == self.W - 1]  # fmt: skip
        idx = [i32([r.slot for r in rows]), i32(positions), i32([r.meta for r in rows]),
               i32(flush + [-1] * (len(rows) - len(flush)))]  # fmt: skip
        return lambda: self.decode_fn(
            mix, a, b, self.al, self.bias, out, self.state, self.dr, self.kr,
            self.gr, *idx, self.sketch, K**-0.5, has_flush_rows=bool(flush),
        )  # fmt: skip

    def window_start(self, r) -> None:
        """The window-start state and the stored coefficient maps."""
        r.s0, r.coeff, r.F = c(self.state[r.slot]), {}, {}
        for h, m in enumerate(self.widths):
            r.coeff[h], r.F[h] = torch.eye(K).double() if m == K else None, []
            if 0 < m < K:
                _, po, _, fg = self.tables.layout[h].tolist()
                p = c(self.sketch.phi[r.meta])
                r.coeff[h] = p[po : po + min(m, 4) * K].reshape(-1, K)
                if m > 4:  # four pivot rows, then the diagonal and the gains
                    ag = p[po + 4 * K : po + 4 * K + 5 * fg].reshape(5, fg)[:, :m]
                    r.coeff[h] = ag[1:].T @ r.coeff[h] + F.pad(ag[0].diag(), (0, K - m))
                err = relative(r.coeff[h], coefficient(r.s0[h], m))
                assert err < 0.008, ("coefficient", m, err)

    def step(self, rows, positions: list[int]) -> None:
        """One decode step of ``rows`` (``PAD`` = padding row), checked."""
        n, H, HV, W, dt = len(rows), self.H, self.HV, self.W, self.dtype
        for r, t in zip(rows, positions):
            if r.slot and t == 0:
                self.window_start(r)
        mix, a, b = self.inputs(n)
        q = unit(c(mix[:, : H * K]).reshape(n, H, K)) / K**0.5
        k = unit(c(mix[:, H * K : 2 * H * K]).reshape(n, H, K))
        v = c(mix[:, 2 * H * K :]).reshape(n, HV, V)
        alpha = torch.exp(-c(self.al).exp() * F.softplus(c(a) + c(self.bias)))
        beta = c(b).sigmoid().to(dt).double()
        dr, kr, gr, br = map(c, (self.dr, self.kr, self.gr, self.sketch.beta))
        expected, exp_d, exp_s = torch.zeros(n, HV, V).double(), {}, {}
        for bi, (r, t) in enumerate(zip(rows, positions)):
            for h, m in enumerate(self.widths if r.slot else []):
                s, s0, kh, qh = r.slot, r.s0[h], k[bi, h // 3], q[bi, h // 3]
                keys, ds, g = kr[s, h // 3, :t], dr[s, h, :t], gr[s, h, :t]
                rep, tot = (g.sum() - g.cumsum(0)).exp(), g.sum().exp()
                kq, kk, ktq = keys @ qh, keys @ kh, kh @ qh
                sq, sk = (rep * kq) @ ds, (rep * kk) @ ds
                at, bt = alpha[bi, h], beta[bi, h]
                if m:  # sketch readout with the projected erase history
                    ff = torch.stack(r.F[h]) if t else torch.zeros(0, m).double()
                    ft = bt * (r.coeff[h] @ kh - kk @ ff)
                    hq = bf(s0[:, :m]) @ (r.coeff[h] @ qh - kq @ ff - ft * ktq)
                    dc = bt * (v[bi, h] - at * sk)
                    r.F[h].append(bf(ft))
                else:  # dense: the BF16 rows of s0
                    hq, hk = bf(s0) @ qh, bf(s0) @ kh
                    dc = bt * (v[bi, h] - at * (tot * hk + sk))
                expected[bi, h], exp_d[s, h] = at * (tot * hq + sq) + dc * ktq, dc
                if t == W - 1:  # exact flush: W erases and updates from s0
                    keys, betas = torch.cat([keys, bf(kh)[None]]), br[r.meta, h]
                    gates, betas[-1] = torch.cat([g, at.log()[None]]), bt
                    # Dense heads stored full updates from bf(s0): erase the rest.
                    st = s0 if m else s0 - bf(s0)
                    for j in range(W):
                        erase = (st @ keys[j])[:, None] * keys[j][None, :]
                        st = gates[j].exp() * (st - betas[j] * erase)
                    if not m:
                        st += bf(s0) * gates.exp().prod()
                    rep = (gates.sum() - gates.cumsum(0)).exp()[:, None]
                    exp_s[s, h] = st + (torch.cat([ds, dc[None]]) * rep).T @ keys
                    expected[bi, h] = exp_s[s, h] @ qh
        out, before = torch.empty(n, HV, V, dtype=dt, device="cuda"), self.state.clone()
        self.decode(mix, a, b, out, rows, positions)()
        err = relative(c(out), expected)
        assert err < (0.008 if dt == torch.bfloat16 else 0.0015), ("output", err)
        flushed = [r.slot for r, t in zip(rows, positions) if r.slot and t == W - 1]
        kept = [i for i in range(NX) if i not in flushed]
        assert torch.equal(self.state[kept], before[kept])
        for bi, (r, t) in enumerate(zip(rows, positions)):
            s = r.slot
            if not s:
                assert not out[bi].any()
            elif t < W - 1:  # the ring rows of this step
                d_ref = bf(torch.stack([exp_d[s, h] for h in range(HV)]))
                assert max(map(relative, c(self.dr[s, :, t]), d_ref)) < 0.004
                assert relative(c(self.kr[s, :, t]), bf(k[bi])) < 0.002
            else:
                target = torch.stack([exp_s[s, h] for h in range(HV)])
                assert relative(c(self.state[s]), target) < 4e-4, "flush"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("window, dtype, widths, degenerate", [
    (16, torch.bfloat16, WIDTHS, False), (32, torch.bfloat16, WIDTHS, False),
    (16, torch.float32, [0, 6, 9], True),  # rank-deficient prefilled states
])  # fmt: skip
def test_gdn_sketch_decode(backend, window, dtype, widths, degenerate):
    """Cold build, then decode at mixed window positions through a flush."""
    hn = Harness(backend, window, dtype, widths, degenerate)
    reqs = [Request(2, 4, 0), Request(6, 0, 5), Request(4, 1, 11)]
    gen = torch.Generator().manual_seed(0)
    for n in range(window + 12):
        if joining := [r for r in reqs if r.start == n]:
            hn.build([*joining, Request(7, 3)], [1] * len(joining) + [0])
            assert not hn.sketch.u[3].any()
        rows = [r for r in reqs if r.start <= n]
        rows += [PAD] * (4 - len(rows))
        rows = [rows[i] for i in torch.randperm(4, generator=gen).tolist()]
        hn.step(rows, [(n - r.start) % window if r.slot else 0 for r in rows])


@pytest.mark.parametrize("backend", BACKENDS)
def test_gdn_dense_rows(backend):
    """Dense heads read their BF16 rows on non-flush steps, not the FP32 state;
    the build and the flush write the rows as the state rounded to BF16."""
    hn = Harness(backend, 16)
    reqs = [Request(2, 4), Request(6, 0)]
    dense = [h for h, m in enumerate(WIDTHS) if m == 0]

    def check_rows():
        for r in reqs:
            for h in dense:
                u_off = int(hn.tables.layout[h, 0])
                rows = hn.sketch.u[r.meta, u_off : u_off + K * V].view(K, V)
                assert torch.equal(rows, hn.state[r.slot, h].T.bfloat16())

    hn.build(reqs, [1, 1])
    check_rows()
    s = hn.sketch
    for t in range(hn.W):
        mix, a, b = hn.inputs(len(reqs))
        out = torch.empty(len(reqs), hn.HV, V, dtype=hn.dtype, device="cuda")
        step = hn.decode(mix, a, b, out, reqs, [t] * len(reqs))
        if t == 3:
            buffers = [hn.state, hn.dr, hn.kr, hn.gr, s.fs, s.beta]
            saved = [x.clone() for x in buffers]
            step()
            expected = out.clone()
            for x, y in zip(buffers, saved):
                x.copy_(y)
            hn.state[:, dense] = hn.state[:, dense] * 3 + 1
            step()
            assert torch.equal(out, expected)
            hn.state.copy_(saved[0])
        else:
            step()
    check_rows()  # rebuilt by the flush at t = W - 1
