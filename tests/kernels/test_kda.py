# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""KDA SketchSSM CUDA decode against an FP64 oracle.

Per request, the oracle solves the four-pivot residual-diagonal coefficient
system of the window-start state with a dense solve, tracks the window's
exact transition and replayed writes in FP64, and flushes by applying the
exact recurrence. Tolerances are the reference check's, for every window
(16, the paper's, and 32, 64, 128); the step counts and check points scale
with the window W (the W = 16 values are the reference check's).
"""

import math
from typing import TypeAlias, cast

import pytest
import torch

pytest.importorskip("vllm")

from vllm.model_executor.layers.mamba.ops.kda_sketchssm_common import (
    KDASketchArgs,
    KDASketchRings,
    KDASketchTables,
    kda_sketch_fold_window_,
    kda_sketch_paged_views,
    kda_sketch_ring_specs,
)

from sketchssm import kernels
from sketchssm.kernels import kda as kd

D = 128
SLOW = pytest.mark.slow
WINDOWS = [16, *(pytest.param(w, marks=SLOW) for w in (32, 64))]


def skip_unsupported(heads: int, window: int) -> None:
    support = kernels.kda_supported(heads, D, D, window, torch.bfloat16, torch.float32)
    if not support:
        pytest.skip(support.reason)


def scratch(max_rows: int, tables: KDASketchTables) -> torch.Tensor:
    numel = kd.scratch_numel(max_rows, tables.num_sketch_heads)
    return torch.empty(numel, dtype=torch.float32, device="cuda")


def state_atol(window: int) -> float:
    """Absolute tolerance of the flushed state against the oracle: the
    reference check's 3e-6 at W = 16; 4e-6 (the graph lifecycle check's) for
    W > 16. The CUDA flush's element errors do not grow with W (the gate's
    error quantiles and mean |error|, 2.0e-7, are the same for W = 16 ... 64)
    but its worst element over the ~0.5M checked is a tail statistic already
    at 0.78 of the bound at W = 16, and W > 16 runs check more flushes (one
    W = 32 gate element: 3.97e-6 at |state| 4.1e-4, ratio 1.16)."""
    return 3e-6 if window == 16 else 4e-6


# Dense (0, 128), the merged small ranks, and every flush rank bucket edge.
RANKS = [0, 1, 2, 3, 4, 5, 6, 7, 12, 16, 17, 28, 33, 60, 65, 127, 128]


def coefficient(state, frame, rank, pivots=4):
    """Oracle map (K x rank) of a window-start state; frame columns
    are the basis."""
    s = state.double() @ frame.double()
    mu = s.square().sum() / 128
    s = s / torch.sqrt(mu if mu > 0 else torch.ones_like(mu))
    q: list[torch.Tensor] = []
    for i in range(min(rank, pivots)):
        u = s[:, i]
        v = u.clone()
        for _ in range(2):
            for b in q:
                v = v - b * (b @ v)
        keep = v.square().sum() > 1e-12 * u.square().sum()
        q.append(v / v.norm() if keep else torch.zeros_like(v))
    z = torch.stack(q) @ s
    residual = (s.square().sum(0) - z.square().sum(0)).clamp_min(0)
    residual[: min(rank, pivots)] = 0
    metric = z.T @ z + torch.diag(residual)
    a = torch.linalg.solve(
        metric[:rank, :rank] + 0.003 * torch.eye(rank, dtype=torch.float64),
        metric[:rank, :],
    )
    return frame.double() @ a.T


class Oracle:
    """FP64 oracle: state pages ``s``, window position per page."""

    def __init__(self, state, frame, ranks, window=16):
        self.window = window
        self.state = state.double().cpu().clone()
        self.frame = frame.double().cpu()
        self.ranks = [0 if m == 128 else m for m in ranks]
        self.start = self.state.clone()
        n, h = state.shape[:2]
        eye = torch.eye(128, dtype=torch.float64)
        self.transition = eye.expand(n, h, 128, 128).clone()
        self.replay = torch.zeros_like(self.state)
        self.pos = [0] * n
        self.maps = {}

    def reset(self, s, state):
        self.state[s] = state.double().cpu()
        self.start[s] = self.state[s]
        self.transition[s] = torch.eye(128, dtype=torch.float64)
        self.replay[s].zero_()
        self.pos[s] = 0

    def step(self, ids, q, k, v, gate, beta, a_log, bias):
        q, k, v, gate, beta, a_log, bias = [
            x.double().cpu() for x in (q, k, v, gate, beta, a_log, bias)
        ]
        bias = bias.reshape(len(self.ranks), D)
        q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) / 128**0.5
        k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        decay = torch.exp(
            -5 * torch.sigmoid(a_log.exp()[None, :, None] * (gate + bias[None]))
        )
        beta = beta.sigmoid()
        out = torch.zeros_like(v)
        for b, s in enumerate(ids):
            if s <= 0:
                continue
            if self.pos[s] == 0:
                for h, m in enumerate(self.ranks):
                    if m:
                        self.maps[s, h] = coefficient(
                            self.start[s, h], self.frame[h], m
                        )
            for h, m in enumerate(self.ranks):
                kh, qh, bt = k[b, h], q[b, h], beta[b, h]
                for mat, write in (
                    (self.state, True),
                    (self.replay, True),
                    (self.transition, False),
                ):
                    x = mat[s, h] * decay[b, h][None, :]
                    delta = -(x @ kh) * bt
                    if write:
                        delta = delta + bt * v[b, h]
                    mat[s, h] = x + delta[:, None] * kh[None, :]
                if m == 0 or self.pos[s] == self.window - 1:
                    out[b, h] = self.state[s, h] @ qh
                else:
                    coef = self.maps[s, h].T @ (self.transition[s, h] @ qh)
                    out[b, h] = (
                        self.start[s, h] @ self.frame[h, :, :m]
                    ) @ coef + self.replay[s, h] @ qh
            self.pos[s] += 1
            if self.pos[s] == self.window:
                self.reset(s, self.state[s])
        return out


def relative(a, b) -> float:
    return float((a.double().cpu() - b).norm() / b.norm().clamp_min(1e-30))


def close_ratio(a, b, rtol=1e-3, atol=3e-6) -> float:
    """Worst ``|a - b| / (atol + rtol |b|)`` (assert_close passes below 1)."""
    return float(((a.double().cpu() - b).abs() / (atol + rtol * b.abs())).max())


class Worst(dict):
    def __call__(self, name: str, value: float) -> float:
        self[name] = max(self.get(name, 0.0), value)
        return value

    def report(self, label: str) -> None:
        print(label, " ".join(f"{k} {v:.2e}" for k, v in self.items()))


# Decode inputs: q, k, v, gate and beta.
Inputs: TypeAlias = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


def paged_cache(
    initial: torch.Tensor, window: int
) -> tuple[torch.Tensor, KDASketchRings]:
    """State and rings in padded pages, as in vLLM's Mamba cache."""
    slots, heads = initial.shape[:2]
    specs: list[tuple[tuple[int, ...], torch.dtype]] = [((heads, D, D), torch.float32)]
    specs += kda_sketch_ring_specs(heads, window).values()
    packed = sum(math.prod(s) * t.itemsize for s, t in specs)
    page = (packed + 511) // 512 * 512 + 1024
    state, *rings = kda_sketch_paged_views(specs, slots, page, "cuda")
    state.copy_(initial)
    return state, KDASketchRings(*rings)


class Harness:
    """vLLM side: state slots (pages) holding the state and the rings,
    persistent request rows (sketch), positions."""

    def __init__(self, initial, frame, ranks, num_reqs, max_batch, window=16):
        skip_unsupported(initial.shape[1], window)
        self.build, self.decode = kernels.kda_cold_build, kernels.kda_decode
        self.window = window
        self.state, self.rings = paged_cache(initial, window)
        self.tables = KDASketchTables(
            frame.transpose(-1, -2).contiguous(), ranks, window
        )
        self.sketch = KDASketchArgs.allocate(self.tables, num_reqs, "cuda")
        self.scratch = scratch(max_batch, self.tables)
        self.req: dict[int, int] = {}
        self.pos: dict[int, int] = {}

    def admit(self, pages: list[int], reqs: list[int]) -> None:
        """Cold build of new requests (pages -> request rows) at window start."""
        n = len(pages)
        dev = self.state.device
        for p, r in zip(pages, reqs):
            self.req[p], self.pos[p] = r, 0
        rows = torch.arange(n, device=dev, dtype=torch.int32)
        rows = torch.cat([rows, torch.full((3,), -1, device=dev, dtype=torch.int32)])
        slots = torch.tensor(pages + [0] * 3, device=dev, dtype=torch.int32)
        meta = torch.tensor(reqs + [0] * 3, device=dev, dtype=torch.int32)
        self.build(
            self.state, self.rings, slots, meta, rows, self.sketch,
            scratch(n + 3, self.tables),
        )  # fmt: skip

    def metadata(self, pages: list[int]):
        dev = self.state.device
        meta = [self.req.get(p, 0) for p in pages]
        pos = [self.pos.get(p, 0) for p in pages]
        last = self.window - 1
        flush = [i for i, p in enumerate(pages) if p > 0 and self.pos[p] == last]
        flush += [-1] * (len(pages) - len(flush))
        as_t = lambda x: torch.tensor(x, device=dev, dtype=torch.int32)  # noqa: E731
        return as_t(pages), as_t(meta), as_t(pos), as_t(flush)

    def advance(self, pages: list[int]) -> None:
        for p in pages:
            if p > 0:
                self.pos[p] = (self.pos[p] + 1) % self.window

    def step(self, pages, q, k, v, gate, beta, a_log, bias, out=None):
        slots, meta, pos, flush = self.metadata(pages)
        out = torch.empty_like(v) if out is None else out
        self.decode(
            q, k, v, gate, beta, a_log, bias, out, self.state, self.rings, slots,
            meta, pos, flush, self.sketch, self.scratch,
        )  # fmt: skip
        self.advance(pages)
        return out

    def phi(self, page, h, m):
        return self.sketch.phi[self.req[page], h, :m].double().cpu().T


def inputs(batch, heads, device, gen_shift=3.0) -> Inputs:
    q, k, v, gate = (
        torch.randn(batch, heads, D, device=device, dtype=torch.bfloat16)
        for _ in range(4)
    )
    gate.sub_(gen_shift)
    beta = torch.randn(batch, heads, device=device, dtype=torch.bfloat16)
    return q, k, v, gate, beta


def setup(ranks, pages, seed):
    torch.manual_seed(seed)
    heads = len(ranks)
    frame = torch.linalg.qr(torch.randn(heads, D, D, dtype=torch.float64)).Q.float()
    initial = torch.randn(pages, heads, D, D) * 0.1
    return frame, initial


@pytest.mark.parametrize("window", WINDOWS)
@torch.inference_mode()
def test_kda_gate(window):
    """Reference check: FP64 oracle outputs, coefficient maps and flushed states
    with a zero state head, an exactly dependent pivot pair, padding, row
    reordering and request rows unrelated to the state slots, over two
    windows."""
    w, atol = window, state_atol(window)
    ranks = torch.tensor(RANKS)
    heads = len(RANKS)
    frame, initial = setup(RANKS, 4, 23)
    initial[:, 1].zero_()
    dependent = initial[:, 7] @ frame[7]
    dependent[:, :, 1] = dependent[:, :, 0]
    initial[:, 7] = dependent @ frame[7].T

    hn = Harness(initial, frame.cuda(), ranks, num_reqs=6, max_batch=3, window=w)
    state = hn.state
    ref = Oracle(initial, frame, RANKS, w)
    a = torch.randn(heads) * 0.1
    bias = torch.randn(heads * D) * 0.1
    pages = [2, 1, 0]
    hn.admit([2, 1], [5, 3])
    worst = Worst()
    for t in range(2 * w + 3):
        if t == w + 3:
            pages = [1, 2, 0]
        data = inputs(3, heads, "cuda")
        out = hn.step(pages, *data, a.cuda(), bias.cuda())
        expected = ref.step(pages, *data, a, bias)
        assert worst("output", relative(out, expected)) < 0.007, t
        assert torch.count_nonzero(out[2]) == 0
        if t in (0, w - 1, 2 * w - 1):
            for page in (1, 2):
                for h, m in enumerate(RANKS):
                    if m in (0, 128):
                        continue
                    want = coefficient(ref.start[page, h], frame[h], m)
                    err = worst("coefficient", relative(hn.phi(page, h, m), want))
                    assert err < 4e-3, (t, m, err)
        if t in (w - 1, 2 * w - 1):
            worst("state", close_ratio(state[1:3], ref.state[1:3], atol=atol))
            torch.testing.assert_close(
                state[1:3].double().cpu(), ref.state[1:3], rtol=1e-3, atol=atol
            )
    # The pending window folded into the state is exact too.
    slots, _, pos, _ = hn.metadata([1, 2])
    kda_sketch_fold_window_(state, hn.rings, slots, pos, hn.tables)
    worst("state", close_ratio(state[1:3], ref.state[1:3], atol=atol))
    torch.testing.assert_close(
        state[1:3].double().cpu(), ref.state[1:3], rtol=1e-3, atol=atol
    )
    worst.report(f"W{w} gate:")


@pytest.mark.parametrize("window", WINDOWS)
@torch.inference_mode()
def test_kda_flush_ignores_corrupted_sketch(window):
    """The flush output and state depend only on the exact rings and state."""
    w, atol = window, state_atol(window)
    ranks = [0, 1, 3, 7, 28]
    frame, initial = setup(ranks, 3, 137)

    hn = Harness(
        initial,
        frame.cuda(),
        torch.tensor(ranks),
        num_reqs=2,
        max_batch=2,
        window=w,
    )
    state = hn.state
    ref = Oracle(initial, frame, ranks, w)
    heads = len(ranks)
    a, bias = torch.zeros(heads), torch.zeros(heads * D)
    hn.admit([1, 2], [1, 0])
    worst = Worst()
    for t in range(2 * w):
        data = inputs(2, heads, "cuda")
        if t in (w // 2 - 1, w + w // 2 - 1):
            hn.sketch.phi.fill_(float("nan"))
            hn.sketch.u.fill_(float("nan"))
        out = hn.step([1, 2], *data, a.cuda(), bias.cuda())
        expected = ref.step([1, 2], *data, a, bias)
        if t in (w - 1, 2 * w - 1):
            assert worst("flush output", relative(out, expected)) < 4e-3
            worst("state", close_ratio(state, ref.state, atol=atol))
            torch.testing.assert_close(
                state.double().cpu(), ref.state, rtol=1e-3, atol=atol
            )
    worst.report(f"W{w} corrupted sketch:")


@pytest.mark.parametrize("window", WINDOWS)
@torch.inference_mode()
def test_kda_mixed_positions(window):
    """Mixed flush / non-flush batches: requests spread over the window's
    positions, rows reordered and padded each step, a request replaced
    mid-run."""
    w, atol = window, state_atol(window)
    every = w // 16  # admissions spread over the first window
    ranks = [1, 2, 4, 5, 9, 16, 17, 40, 0, 70]
    heads, pages, batch = len(ranks), 20, 18
    frame, initial = setup(ranks, pages, 7)

    hn = Harness(
        initial,
        frame.cuda(),
        torch.tensor(ranks),
        num_reqs=24,
        max_batch=batch,
        window=w,
    )
    state = hn.state
    ref = Oracle(initial, frame, ranks, w)
    a = torch.randn(heads) * 0.2
    bias = torch.randn(heads * D) * 0.1
    live: list[int] = []
    gen = torch.Generator().manual_seed(3)
    worst = Worst()
    for t in range(w + 24):
        if t < w and t % every == 0:
            i = t // every
            new = [1 + i, 17] if i == 0 else [1 + i]
            hn.admit(new, [23 - p for p in new])
            live += new
        if t == w + 11:
            gone = live.pop(4)
            state[gone].normal_(std=0.1)
            ref.reset(gone, state[gone])
            hn.admit([gone], [0])
            live.append(gone)
        perm = torch.randperm(len(live), generator=gen).tolist()
        rows = [live[i] for i in perm]
        rows.insert(t % len(rows), 0)
        rows = rows[:batch] + [0] * (batch - len(rows))
        data = inputs(batch, heads, "cuda")
        out = hn.step(rows, *data, a.cuda(), bias.cuda())
        expected = ref.step(rows, *data, a, bias)
        for r, p in enumerate(rows):
            if p <= 0:
                assert torch.count_nonzero(out[r]) == 0
            else:
                err = worst("output", relative(out[r], expected[r]))
                assert err < 0.007, (t, r, p, ref.pos[p], err)
                if w > 16 and ref.pos[p] == 0:  # flushed this step
                    worst("state", close_ratio(state[p], ref.state[p], atol=atol))
                    torch.testing.assert_close(
                        state[p].double().cpu(), ref.state[p], rtol=1e-3, atol=atol
                    )
    for p in live:
        if ref.pos[p] == 0:
            worst("state", close_ratio(state[p], ref.state[p], atol=atol))
            torch.testing.assert_close(
                state[p].double().cpu(), ref.state[p], rtol=1e-3, atol=atol
            )
    worst.report(f"W{w} mixed:")


@pytest.mark.parametrize("window", WINDOWS)
@torch.inference_mode()
def test_kda_cuda_graph_lifecycle(window):
    """One captured decode replayed over many windows with changing rows,
    pads and flush lists matches eager and keeps the state exact."""
    w = window
    ranks = [7, 28, 0, 1]
    heads, pages, batch = len(ranks), 12, 8
    frame, initial = setup(ranks, pages, 300)

    hn = Harness(
        initial,
        frame.cuda(),
        torch.tensor(ranks),
        num_reqs=10,
        max_batch=batch,
        window=w,
    )

    eager = Harness(initial, frame.cuda(), torch.tensor(ranks), 10, batch, w)
    state, eager_state = hn.state, eager.state
    a, bias = torch.zeros(heads, device="cuda"), torch.zeros(heads * D, device="cuda")
    live = [1, 2, 3, 4, 5, 6, 7]
    reqs = [9, 0, 4, 6, 2, 1, 8]
    hn.admit(live, reqs)
    eager.admit(live, reqs)
    data = inputs(batch, heads, "cuda")
    static = cast(Inputs, tuple(x.clone() for x in data))
    meta_bufs = [torch.zeros(batch, device="cuda", dtype=torch.int32) for _ in range(4)]
    meta_bufs[3].fill_(-1)
    out = torch.empty_like(data[2])
    slots_buf, meta_buf, pos_buf, flush_buf = meta_bufs

    def launch():
        hn.decode(
            *static,
            a,
            bias,
            out,
            state,
            hn.rings,
            slots_buf,
            meta_buf,
            pos_buf,
            flush_buf,
            hn.sketch,
            hn.scratch,
        )

    hn_state = state.clone()
    launch()  # compile before capture, then restore
    state.copy_(hn_state)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    state.copy_(hn_state)
    oracle = Oracle(initial, frame, ranks, w)
    for step in range(4 * w + 6):
        rows = list(live) if step % 2 else list(reversed(live))
        rows.insert(step % 8, 0)
        rows = rows[:batch]
        data = inputs(batch, heads, "cuda")
        for s, x in zip(static, data):
            s.copy_(x)
        for buf, x in zip(meta_bufs, hn.metadata(rows)):
            buf.copy_(x)
        hn.advance(rows)
        graph.replay()
        expected = eager.step(rows, *data, a, bias)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        torch.testing.assert_close(state, eager_state, rtol=0, atol=0)
        for x, y in zip(hn.rings.tensors(), eager.rings.tensors()):
            assert same_bits(x, y)
        oracle.step(rows, *data, a.cpu(), bias.cpu())
    slots, _, pos, _ = hn.metadata(live)
    kda_sketch_fold_window_(state, hn.rings, slots, pos, hn.tables)
    ratio = close_ratio(state[1:8], oracle.state[1:8], atol=4e-6)
    print(f"W{w} graph lifecycle: state {ratio:.2e}")
    torch.testing.assert_close(
        state[1:8].double().cpu(), oracle.state[1:8], rtol=1e-3, atol=4e-6
    )


def same_bits(x: torch.Tensor, y: torch.Tensor) -> bool:
    """Bitwise equality: dense heads keep FP32 replay rows in the bytes of
    their BF16 u / d rings, which may read as BF16 NaNs."""
    ints = {2: torch.int16, 4: torch.int32}
    return torch.equal(x.view(ints[x.element_size()]), y.view(ints[y.element_size()]))


def fused_views(q, k, v, gate, beta) -> Inputs:
    """The decode inputs as the model passes them: q, k, v and beta are
    row-strided views of one fused projection row, the gate a row-padded
    view."""
    batch, heads, _ = q.shape
    like = {"device": q.device, "dtype": q.dtype}
    proj = torch.empty(batch, 3 * heads * D + heads + 2 * D, **like)
    qkv, beta_v, _, _ = proj.split([3 * heads * D, heads, D, D], dim=-1)
    q_v, k_v, v_v = (x.view(batch, heads, D) for x in qkv.split(heads * D, -1))
    gate_v = torch.empty(batch, heads * D + 64, **like)[:, 32 : 32 + heads * D]
    views = (q_v, k_v, v_v, gate_v.view(batch, heads, D), beta_v)
    for view, x in zip(views, (q, k, v, gate, beta)):
        view.copy_(x)
    return views


@pytest.mark.parametrize("window", WINDOWS)
@torch.inference_mode()
def test_kda_strided_inputs(window):
    """Row-strided q / k / v / gate / beta views (read in place) give the same
    bits as contiguous inputs: outputs, state, rings and sketch, over flush
    and non-flush rows."""
    w = window
    ranks = [0, 3, 17, 70]
    heads, pages, batch = len(ranks), 10, 8
    frame, initial = setup(ranks, pages, 11)
    args = (initial, frame.cuda(), torch.tensor(ranks), 10, batch, w)
    dense, strided = Harness(*args), Harness(*args)
    a = torch.randn(heads, device="cuda") * 0.2
    bias = torch.randn(heads * D, device="cuda") * 0.1
    for t in range(w + 4):
        if t < 6:
            for hn in (dense, strided):
                hn.admit([1 + t], [9 - t])
        rows = [p for p in range(1, 7) if p <= t + 1][::-1] + [0]
        rows = rows[:batch] + [0] * (batch - len(rows))
        data = inputs(batch, heads, "cuda")
        views = fused_views(*data)
        assert not any(x.is_contiguous() for x in views)
        out = dense.step(rows, *data, a, bias)
        torch.testing.assert_close(
            strided.step(rows, *views, a, bias), out, rtol=0, atol=0
        )

    def buffers(hn):
        sketch = (hn.sketch.u, hn.sketch.phi, hn.sketch.f, hn.sketch.dense)
        return [hn.state, *hn.rings.tensors(), *sketch]

    for x, y in zip(buffers(dense), buffers(strided)):
        assert same_bits(x, y)


@pytest.mark.parametrize(
    "window", [16, *(pytest.param(w, marks=SLOW) for w in (32, 64))]
)
@torch.inference_mode()
def test_kda_strong_decay(window):
    """Half the key channels decay near the gate's lower bound every step, so
    their cumulative log decay reaches about -4.7 W: the window's decayed
    keys must stay finite (rebased per 8-row chunk in the step and per
    16-row block in the flush) and the outputs and flushed state exact."""
    w, atol = window, state_atol(window)
    ranks = [0, 3, 17, 70]
    heads = len(ranks)
    frame, initial = setup(ranks, 3, 41)
    hn = Harness(initial, frame.cuda(), torch.tensor(ranks), 2, 2, w)
    ref = Oracle(initial, frame, ranks, w)
    a, bias = torch.zeros(heads), torch.zeros(heads * D)
    hn.admit([1, 2], [1, 0])
    worst = Worst()
    for t in range(2 * w):
        q, k, v, gate, beta = inputs(2, heads, "cuda")
        gate[..., : D // 2] += 6.0  # -5 sigmoid(g + 3): about -4.7 per step
        data = (q, k, v, gate, beta)
        out = hn.step([1, 2], *data, a.cuda(), bias.cuda())
        expected = ref.step([1, 2], *data, a, bias)
        assert torch.isfinite(out).all(), t
        assert worst("output", relative(out, expected)) < 0.007, t
        if t in (w - 1, 2 * w - 1):
            worst("state", close_ratio(hn.state[1:3], ref.state[1:3], atol=atol))
            torch.testing.assert_close(
                hn.state[1:3].double().cpu(), ref.state[1:3], rtol=1e-3, atol=atol
            )
    worst.report(f"W{w} strong decay:")


def test_kda_config_lookup(config_dir, clear_caches):
    """Tuned knobs of a window: its own file, else the shape's window-16 file
    on this GPU, else the architecture default."""
    heads = 3
    assert "window" not in kd.config_file_name(heads)
    assert ",window=32," in kd.config_file_name(heads, 32)
    clear_caches(kd.tuned_config)
    assert kd.tuned_config(heads, 32) == kd.default_config(heads, 32)
    (config_dir / kd.config_file_name(heads)).write_text('{"flush": {"K1_MINB": 1}}')
    kd.tuned_config.cache_clear()
    assert kd.tuned_config(heads, 32)[1]["K1_MINB"] == 1
    name = kd.config_file_name(heads, 32)
    (config_dir / name).write_text('{"flush": {"K2_MINB": 4}}')
    kd.tuned_config.cache_clear()
    _, flush = kd.tuned_config(heads, 32)
    assert flush["K2_MINB"] == 4
    assert flush["K1_MINB"] == kd.default_config(heads, 32)[1]["K1_MINB"]
    assert kd.tuned_config(heads, 64)[1]["K1_MINB"] == 1


def test_kda_borrowed_knobs_fallback(config_dir, clear_caches):
    """Window-16 step knobs whose shared memory does not fit at a larger window
    (5 warps of the W = 64 step exceed the 48 KB static limit) fall back to
    the architecture default instead of failing."""
    heads = 5
    skip_unsupported(heads, 64)
    (config_dir / kd.config_file_name(heads)).write_text('{"step": {"NW": 5}}')
    clear_caches(kd.tuned_config, kd._step_ext)
    assert kd.tuned_config(heads, 64)[0]["NW"] == 5
    assert kd._step_ext(heads, 64) is not None
