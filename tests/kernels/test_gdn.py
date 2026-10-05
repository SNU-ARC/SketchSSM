# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""GDN SketchSSM CUDA decode against an FP64 oracle.

Per request, the oracle solves the four-pivot residual-diagonal coefficient
system of the window-start state directly, tracks the projected erase history,
and applies each flush as the exact sequence of delta-rule erases and updates.
Dense heads read the BF16 rows of the window-start state; their flush erases
the rounding residual of those rows.
Each step reads the kernels' own BF16 ring history, so every step is checked
against the exact arithmetic of its inputs. Tolerances are the reference check's.
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("vllm")

from vllm.model_executor.layers.mamba.ops.gdn_sketchssm_common import (
    GDNSketchArgs,
    GDNSketchTables,
    gdn_sketch_build,
)

from sketchssm import kernels

K = V = 128
W = 16  # the paper's window
NX = 8  # state slots; 0 is the null block
NR = 5  # sketch rows

WIDTHS_ALL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 44, 52, 78, 128]
WIDTHS_SKETCHED = [1, 2, 3, 4, 7, 20, 44, 52, 78, 128, 1, 8]
# Value heads per key head: the paper's 3, and Qwen3.5 / Qwen3-Next style 1, 2.
RATIOS = [3, 1, 2, 4]
# Longer windows (multiples of 16), at the paper's and Qwen3.5's value heads per
# key head.
WINDOWS = [32, 64]
WINDOW_RATIOS = [3, 1]


def fit(widths: list[int], hpg: int) -> list[int]:
    """``widths`` extended cyclically to whole groups of ``hpg`` value heads."""
    return [widths[i % len(widths)] for i in range(-(-len(widths) // hpg) * hpg)]


def coefficient(s: torch.Tensor, m: int) -> torch.Tensor | None:
    """FP64 four-pivot residual-diagonal coefficient map (ridge 0.003)."""
    if m == 0:
        return None
    if m == K:
        return torch.eye(K, dtype=torch.float64)
    mean = s.square().sum(0).mean()
    s = s / torch.sqrt(mean if mean > 0 else torch.ones_like(mean))
    ds: list[torch.Tensor] = []
    for j in range(min(m, 4)):
        u = s[:, j]
        v = u.clone()
        for _ in range(2):
            for d in ds:
                v -= d * (d @ v)
        keep = v.square().sum() > 1e-12 * u.square().sum()
        ds.append(v / v.norm() if keep else torch.zeros_like(v))
    z = torch.stack(ds) @ s
    res = (s.square().sum(0) - z.square().sum(0)).clamp_min(0)
    res[: min(m, 4)] = 0
    gram = z.T @ z + torch.diag(res)
    eye = torch.eye(m, dtype=torch.float64)
    return torch.linalg.solve(gram[:m, :m] + 0.003 * eye, gram[:m])


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def stored_coefficient(tables, phi_row, h, m) -> torch.Tensor:
    """Coefficient map ``(m, K)`` held in a request's packed ``phi``."""
    _, po, _, fg = tables.layout[h].tolist()
    p = phi_row.double().cpu()
    if m <= 4:
        return p[po : po + m * K].reshape(m, K)
    tail = p[po : po + 4 * K].reshape(4, K)
    ag = p[po + 4 * K : po + 4 * K + 5 * fg].reshape(5, fg)[:, :m]
    out = ag[1:].T @ tail
    out[:, :m] += torch.diag(ag[0])
    return out


class Request:
    def __init__(self, slot: int, meta: int, start: int):
        self.slot, self.meta, self.start = slot, meta, start
        self.s0: torch.Tensor | None = None
        self.coeff: dict[int, torch.Tensor | None] = {}
        self.F: dict[int, list[torch.Tensor]] = {}


class Harness:
    """One GDN layer, its requests and the oracle's bookkeeping."""

    def __init__(self, dtype, widths, degenerate=False, seed=17, hpg=3, window=W):
        torch.manual_seed(seed)
        assert len(widths) % hpg == 0
        self.dtype, self.widths, self.hpg, self.W = dtype, widths, hpg, window
        self.HV = len(widths)
        self.H = self.HV // hpg
        support = kernels.gdn_supported(
            self.H, self.HV, K, V, window, torch.bfloat16, torch.float32
        )
        if not support:
            pytest.skip(support.reason)
        state = torch.randn(NX, self.HV, V, K, device="cuda") * 0.12
        if degenerate:
            state[:, :, :, 0] = 0
            state[:, :, :, 2] = state[:, :, :, 1]
            state[:, 2] = 0
        self.state = state
        with torch.device("cuda"):
            self.tables = GDNSketchTables(torch.tensor(widths), self.H, window)
        self.sketch = GDNSketchArgs.allocate(self.tables, NR, "cuda")
        nan = float("nan")
        self.dr = torch.full((NX, self.HV, window, V), nan, device="cuda").bfloat16()
        self.kr = torch.full((NX, self.H, window, K), nan, device="cuda").bfloat16()
        self.gr = torch.zeros(NX, self.HV, window, device="cuda")
        self.al = torch.randn(self.HV, device="cuda") * 0.05
        self.bias = torch.zeros(self.HV, device="cuda")
        self.maxout = self.maxflush = self.maxmap = self.maxring = 0.0

    def build(self, reqs: list[Request]) -> None:
        dev = "cuda"
        slots = torch.tensor([r.slot for r in reqs], device=dev, dtype=torch.int32)
        meta = torch.tensor([r.meta for r in reqs], device=dev, dtype=torch.int32)
        flags = torch.ones(len(reqs), device=dev, dtype=torch.int32)
        gdn_sketch_build(self.state, flags, slots, meta, self.sketch)

    def window_start(self, r: Request) -> None:
        """Window-start bookkeeping: state, coefficient maps, erase history."""
        r.s0 = self.state[r.slot].double().cpu()
        for h, m in enumerate(self.widths):
            r.coeff[h] = coefficient(r.s0[h], m)
            r.F[h] = []
            if m not in (0, K):
                actual = stored_coefficient(self.tables, self.sketch.phi[r.meta], h, m)
                err = relative(actual, r.coeff[h])
                self.maxmap = max(self.maxmap, err)
                assert err < 0.008, ("coefficient", m, err)
                r.coeff[h] = actual

    def inputs(self, batch):
        H, HV, dt = self.H, self.HV, self.dtype
        mix = (torch.randn(batch, 2 * H * K + HV * V, device="cuda") * 0.25).to(dt)
        a = (-3 + torch.randn(batch, HV, device="cuda") * 0.2).to(dt)
        b = (torch.randn(batch, HV, device="cuda") * 0.5).to(dt)
        return mix, a, b

    def decode(self, mix, a, b, out, slots, pos, meta, rows, has_flush=True):
        kernels.gdn_decode(
            mix, a, b, self.al, self.bias, out, self.state, self.dr, self.kr,
            self.gr, slots, pos, meta, rows, self.sketch, K**-0.5,
            has_flush_rows=has_flush,
        )  # fmt: skip

    def step(self, rows: list[Request | None], positions: list[int]) -> None:
        """One decode step of ``rows`` (None = padding row), checked."""
        batch = len(rows)
        H, HV, dt, W = self.H, self.HV, self.dtype, self.W
        for r, p in zip(rows, positions):
            if r is not None and p == 0:
                self.window_start(r)
        mix, a, b = self.inputs(batch)
        q = mix[:, : H * K].double().cpu().reshape(batch, H, K)
        q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) / K**0.5
        k = mix[:, H * K : 2 * H * K].double().cpu().reshape(batch, H, K)
        k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        v = mix[:, 2 * H * K :].double().cpu().reshape(batch, HV, V)
        al, bias = self.al.double().cpu(), self.bias.double().cpu()
        alpha = torch.exp(-al.exp() * F.softplus(a.double().cpu() + bias))
        beta = b.double().cpu().sigmoid()
        dr, kr, gr = (x.double().cpu() for x in (self.dr, self.kr, self.gr))
        br = self.sketch.beta.double().cpu()
        expected = torch.zeros(batch, HV, V, dtype=torch.float64)
        expected_d, expected_state = {}, {}
        for bi, (r, t) in enumerate(zip(rows, positions)):
            if r is None:
                continue
            s, s0 = r.slot, r.s0
            assert s0 is not None
            for h, m in enumerate(self.widths):
                kh, qh = k[bi, h // self.hpg], q[bi, h // self.hpg]
                keys, ds, g = kr[s, h // self.hpg, :t], dr[s, h, :t], gr[s, h, :t]
                pre = g.cumsum(0)
                one = torch.tensor(1.0, dtype=torch.float64)
                tot = pre[-1].exp() if t else one
                rep = (pre[-1] - pre).exp() if t else pre
                kq, kk = keys @ qh, keys @ kh
                sq, sk = (rep * kq) @ ds, (rep * kk) @ ds
                ktq = kh @ qh
                at, bt = alpha[bi, h], beta[bi, h]
                if m:
                    ff = (
                        torch.stack(r.F[h])
                        if t
                        else torch.empty(0, m, dtype=torch.float64)
                    )
                    ft = bt * (r.coeff[h] @ kh - kk @ ff)
                    x = r.coeff[h] @ qh - kq @ ff - ft * ktq
                    hq = s0[h, :, :m].to(torch.bfloat16).double() @ x
                    dc = bt * (v[bi, h] - at * sk)
                    r.F[h].append(ft.to(torch.bfloat16).double())
                else:  # dense: the BF16 rows of s0
                    s0b = s0[h].to(torch.bfloat16).double()
                    hq = s0b @ qh
                    dc = bt * (v[bi, h] - at * (tot * (s0b @ kh) + sk))
                expected[bi, h] = at * (tot * hq + sq) + dc * ktq
                expected_d[s, h] = dc
                if t == W - 1:
                    keys = torch.cat([keys, kh.to(torch.bfloat16).double()[None]], 0)
                    ds = torch.cat([ds, dc[None]], 0)
                    gates = torch.cat([g, at.log()[None]], 0)
                    alphas = gates.exp()
                    # Dense heads stored full updates from their BF16 rows:
                    # only the rounding residual is erased.
                    s0b = s0[h].to(torch.bfloat16).double()
                    boundary = s0[h].clone() if m else s0[h] - s0b
                    betas = torch.cat([br[r.meta, h, :t], bt[None]], 0)
                    for st in range(W):
                        erase = (boundary @ keys[st])[:, None] * keys[st][None, :]
                        boundary = alphas[st] * (boundary - betas[st] * erase)
                    if not m:
                        boundary = boundary + s0b * alphas.prod()
                    pre = gates.cumsum(0)
                    replay = (pre[-1] - pre).exp()
                    expected_state[s, h] = boundary + (ds * replay[:, None]).T @ keys
                    expected[bi, h] = expected_state[s, h] @ qh
        dev = "cuda"
        slots = [r.slot if r else 0 for r in rows]
        metas = [r.meta if r else 0 for r in rows]
        flush = [i for i, (r, p) in enumerate(zip(rows, positions)) if r and p == W - 1]
        flush_rows = flush + [-1] * (batch - len(flush))
        out = torch.empty(batch, HV, V, device=dev, dtype=dt)
        before = self.state.clone()
        self.decode(
            mix, a, b, out,
            torch.tensor(slots, device=dev, dtype=torch.int32),
            torch.tensor(positions, device=dev, dtype=torch.int32),
            torch.tensor(metas, device=dev, dtype=torch.int32),
            torch.tensor(flush_rows, device=dev, dtype=torch.int32),
            has_flush=bool(flush),
        )  # fmt: skip
        err = relative(out.double().cpu(), expected)
        self.maxout = max(self.maxout, err)
        assert err < (0.008 if dt == torch.bfloat16 else 0.0015), ("output", err)
        for bi, r in enumerate(rows):
            if r is None:
                assert torch.count_nonzero(out[bi]) == 0
        for bi, (r, t) in enumerate(zip(rows, positions)):
            if r is None:
                continue
            s = r.slot
            if t < W - 1:
                assert torch.equal(self.state[s], before[s]), "non-flush wrote state"
                for h in range(HV):
                    d_ref = expected_d[s, h].to(torch.bfloat16).double()
                    err = relative(self.dr[s, h, t].double().cpu(), d_ref)
                    self.maxring = max(self.maxring, err)
                    assert err < 0.004, ("ring", s, h, t, err)
                k_ref = k[bi].to(torch.bfloat16).double()
                assert relative(self.kr[s, :, t].double().cpu(), k_ref) < 0.002
            else:
                target = torch.stack([expected_state[s, h] for h in range(HV)])
                err = relative(self.state[s].double().cpu(), target)
                self.maxflush = max(self.maxflush, err)
                assert err < 4e-4, ("flush", s, err)
        untouched = [i for i in range(NX) if i not in slots or i == 0]
        assert torch.equal(self.state[untouched], before[untouched])

    def check_graphs(self, rows: list[Request | None]) -> None:
        """Eager and CUDA-graph replays of the same step agree."""
        dev = "cuda"
        batch = len(rows)
        self.dr.nan_to_num_()
        self.kr.nan_to_num_()
        slots = torch.tensor([r.slot if r else 0 for r in rows], device=dev)
        metas = torch.tensor([r.meta if r else 0 for r in rows], device=dev)
        slots, metas = slots.int(), metas.int()
        for position in (0, self.W // 2 - 1, self.W - 1):
            pos = torch.full((batch,), position, device=dev, dtype=torch.int32)
            flush = [i for i, r in enumerate(rows) if r and position == self.W - 1]
            flush_rows = torch.tensor(
                flush + [-1] * (batch - len(flush)), device=dev, dtype=torch.int32
            )
            mix, a, b = self.inputs(batch)
            out = torch.zeros(batch, self.HV, V, device=dev, dtype=self.dtype)
            s = self.sketch
            buffers = [self.state, self.dr, self.kr, self.gr, s.u, s.phi, s.fs,
                       s.beta, s.current_d, s.current_k]  # fmt: skip

            def execute():
                self.decode(mix, a, b, out, slots, pos, metas, flush_rows)  # noqa: B023

            saved = [x.clone() for x in buffers]
            execute()
            expected = [x.clone() for x in [*buffers, out]]
            for x, y in zip(buffers, saved):
                x.copy_(y)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                execute()
            for x, y in zip(buffers, saved):
                x.copy_(y)
            graph.replay()
            torch.cuda.synchronize()
            for x, y in zip([*buffers, out], expected):
                torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6, equal_nan=True)


GATE_CASES = [
    (WIDTHS_ALL, False),
    (WIDTHS_SKETCHED, False),
    ([0, 6, 9], True),
    ([0, 3, 44], False),
]


@pytest.mark.parametrize("hpg", RATIOS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("widths, degenerate", GATE_CASES)
def test_gdn_gate(dtype, widths, degenerate, hpg):
    """The reference check: three windows of two requests and a padding row,
    with the first request moved to another slot and sketch row (and cold
    built again) at the start of the second window, and the batch rows
    reordered in the third. ``hpg`` value heads share a key head."""
    if degenerate and dtype == torch.bfloat16:
        pytest.skip("the reference check runs the degenerate case in FP32 only")
    run_gate(dtype, widths, degenerate, hpg, W)


@pytest.mark.slow
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("hpg", WINDOW_RATIOS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("widths, degenerate", [(WIDTHS_ALL, False), ([0, 6, 9], True)])
def test_gdn_gate_window(dtype, widths, degenerate, hpg, window):
    """The reference check (same tolerances) at windows longer than 16: the
    flush folds W rows (a W x W WY system) and the step replays up to W - 1."""
    if degenerate and dtype == torch.bfloat16:
        pytest.skip("the reference check runs the degenerate case in FP32 only")
    run_gate(dtype, widths, degenerate, hpg, window)


def run_gate(dtype, widths, degenerate, hpg, window):
    widths = fit(widths, hpg)
    hn = Harness(dtype, widths, degenerate, hpg=hpg, window=window)
    W = window
    r0, r1 = Request(3, 1, 0), Request(1, 2, 0)
    hn.build([r0, r1])
    rows: list[Request | None] = [r0, r1, None]
    for win in range(3):
        if win == 1:
            hn.state[5].copy_(hn.state[3])
            r0.slot, r0.meta = 5, 3
            hn.build([r0])
        if win == 2:
            rows = [None, r1, r0]
        for t in range(W):
            hn.step(rows, [t] * len(rows))
    hn.check_graphs(rows)
    print(
        f"{dtype} W={W} hpg={hpg} {widths}: output {hn.maxout:.2e} "
        f"flush {hn.maxflush:.2e} coefficient {hn.maxmap:.2e} ring {hn.maxring:.2e}"
    )


@pytest.mark.parametrize(
    "hpg, window",
    [
        *((hpg, W) for hpg in [*RATIOS, 5, 6, 7, 8]),
        *(
            pytest.param(hpg, w, marks=pytest.mark.slow)
            for w in WINDOWS
            for hpg in WINDOW_RATIOS
        ),
        pytest.param(3, 48, marks=pytest.mark.slow),  # not a power of two
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gdn_mixed_positions(dtype, hpg, window):
    """Requests at different window positions share every batch, so flush and
    non-flush rows mix; rows are shuffled every step, padding rows included,
    and requests are cold built when they join. Then the eager and CUDA-graph
    runs of a step agree."""
    W = window
    hn = Harness(dtype, fit(WIDTHS_SKETCHED, hpg), seed=3, hpg=hpg, window=window)
    reqs = [Request(2, 4, 0), Request(6, 0, 5), Request(4, 1, 11), Request(7, 2, 14)]
    batch = 6
    gen = torch.Generator().manual_seed(0)
    for n in range(2 * W + 15):
        joining = [r for r in reqs if r.start == n]
        if joining:
            hn.build(joining)
        active = [r for r in reqs if r.start <= n]
        rows: list[Request | None] = [*active] + [None] * (batch - len(active))
        rows = [rows[i] for i in torch.randperm(batch, generator=gen).tolist()]
        positions = [(n - r.start) % W if r else 0 for r in rows]
        hn.step(rows, positions)
    if window != 16:
        hn.check_graphs(rows)
    print(
        f"mixed W={W} hpg={hpg}: output {hn.maxout:.2e} "
        f"flush {hn.maxflush:.2e} "
        f"coefficient {hn.maxmap:.2e} ring {hn.maxring:.2e}"
    )
