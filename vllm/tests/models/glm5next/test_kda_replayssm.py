# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA ReplaySSM (exact W=16 window) decode.

Every output and every flushed native page must match the per-step recurrence
(``fused_recurrent_kda``) across slot reorder, eviction, release, prefill
handoff and mixed prefill + decode steps.
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.kda_replayssm import KDAReplaySSM
from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only Gluon/Triton kernels"
)

OUT_TOL = dict(rtol=0.008, atol=1e-4)
STATE_TOL = dict(rtol=3e-5, atol=2e-6)


def _inputs(batch: int, heads: int, gate_shift: float = 4.0) -> list[torch.Tensor]:
    data = [
        torch.randn(batch, heads, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    ]
    data[3].sub_(gate_shift)
    data.append(torch.randn(batch, heads, device="cuda", dtype=torch.bfloat16))
    return data


def _recurrent(state, ids, data, a, bias) -> torch.Tensor:
    """Exact per-step reference; updates ``state`` pages in place."""
    out, _ = fused_recurrent_kda(
        q=data[0].unsqueeze(0),
        k=data[1].unsqueeze(0),
        v=data[2].unsqueeze(0),
        g=data[3].unsqueeze(0),
        beta=data[4].unsqueeze(0),
        a_log=a,
        g_bias=bias,
        initial_state=state,
        cu_seqlens=torch.arange(ids.numel() + 1, device="cuda", dtype=torch.int32),
        ssm_state_indices=ids,
        use_qk_l2norm_in_kernel=True,
        sigmoid_beta=True,
        compute_gate=True,
        lower_bound=-5.0,
    )
    return out[0]


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.double() - expected.double()).abs()
    rel = diff.norm() / expected.double().norm().clamp_min(1e-30)
    return diff.max().item(), rel.item()


@torch.inference_mode()
def test_replay_matches_recurrence_with_staggered_requests():
    """Requests join at different window offsets, rows are shuffled every
    step and each request crosses at least four flushes."""
    torch.manual_seed(11)
    heads, pages = 4, 9
    state = torch.randn(pages, heads, 128, 128, device="cuda") * 0.1
    expected = state.clone()
    cache = KDAReplaySSM(heads, 8, state.device)
    a = 0.5 * torch.randn(heads, device="cuda")
    bias = 0.1 * torch.randn(heads, 128, device="cuda")
    starts = {1: 0, 2: 3, 3: 7, 4: 11, 5: 13, 6: 21}
    out_err, state_err = [0.0, 0.0], [0.0, 0.0]
    for t in range(90):
        active = [p for p, s in starts.items() if s <= t]
        order = torch.randperm(len(active)).tolist()
        rows = [active[i] for i in order] + [0]
        ids = torch.tensor(rows, device="cuda", dtype=torch.int32)
        data = _inputs(len(rows), heads)
        out = cache.step(state, ids, *data, a, bias)
        ref = _recurrent(expected, ids, data, a, bias)
        torch.testing.assert_close(out[:-1], ref[:-1], **OUT_TOL)
        assert torch.count_nonzero(out[-1]) == 0
        out_err = [max(x, y) for x, y in zip(out_err, _errors(out[:-1], ref[:-1]))]
        for page in active:
            if (t - starts[page]) % 16 == 15:
                torch.testing.assert_close(state[page], expected[page], **STATE_TOL)
                err = _errors(state[page], expected[page])
                state_err = [max(x, y) for x, y in zip(state_err, err)]
    assert min(90 - s for s in starts.values()) >= 4 * 16
    all_ids = torch.arange(1, pages, device="cuda", dtype=torch.int32)
    cache.before_prefill(state, all_ids, torch.ones_like(all_ids, dtype=torch.bool))
    torch.testing.assert_close(state, expected, **STATE_TOL)
    print(
        f"\nstaggered: out max abs {out_err[0]:.3e} rel {out_err[1]:.3e}; "
        f"flushed state max abs {state_err[0]:.3e} rel {state_err[1]:.3e}; "
        f"final handoff {_errors(state, expected)}"
    )


@pytest.mark.parametrize("graph", [False, True])
@torch.inference_mode()
def test_replay_survives_reorder_eviction_release_and_prefill(graph):
    """Physical pages may outlive slots; freed pages must never be flushed."""
    torch.manual_seed(217)
    heads, capacity, batch = 3, 4, 3
    state = torch.empty_strided(
        (9, heads, 128, 128),
        (heads * 16384 + 256, 16384, 128, 1),
        device="cuda",
        dtype=torch.float32,
    )
    state.normal_(std=0.1)
    expected_state = state.clone()
    cache = KDAReplaySSM(heads, capacity, state.device)
    ids = torch.tensor([1, 2, 0], dtype=torch.int32, device="cuda")
    data = _inputs(batch, heads)
    a = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads, 128, device="cuda")

    def execute():
        return cache.step(state, ids, *data, a, bias)

    execute()  # Compile before capture; no live request state is retained.
    cache.reset()
    state.copy_(expected_state)
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = execute()
        cache.reset()
        state.copy_(expected_state)

    for t in range(257):
        # More physical pages than slots exercises eviction of absent owners.
        active = [1 + (t // 19) % 7, 1 + (t // 19 + 3) % 7, 0]
        if t % 2:
            active[:2] = active[:2][::-1]
        ids.copy_(torch.tensor(active, device="cuda", dtype=torch.int32))
        if t % 29 == 0:
            cache.release_finished(ids[:1])
            state[active[0]].zero_()
            expected_state[active[0]].zero_()
        if t % 11 == 0:
            cache.before_prefill(
                state, ids[:1], torch.ones(1, device="cuda", dtype=torch.bool)
            )
            torch.testing.assert_close(
                state[active[0]], expected_state[active[0]], **STATE_TOL
            )
        data[0].normal_()
        if graph:
            g.replay()
        else:
            out = execute()
        native = _recurrent(expected_state, ids, data, a, bias)
        torch.testing.assert_close(out[:2], native[:2], **OUT_TOL)
        assert torch.count_nonzero(out[2]) == 0
    all_ids = torch.arange(1, 9, device="cuda", dtype=torch.int32)
    cache.before_prefill(state, all_ids, torch.ones(8, device="cuda", dtype=torch.bool))
    torch.testing.assert_close(state, expected_state, **STATE_TOL)


@torch.inference_mode()
def test_replay_varied_inputs_and_channel_decay_fp64():
    """Parallel factors keep the erase cross terms and per-channel decay; the
    native page is only written on the flush step."""
    torch.manual_seed(491)
    h = 3
    state = torch.randn(2, h, 128, 128, device="cuda") * 0.1
    reference = state[1].double()
    cache = KDAReplaySSM(h, 1, state.device)
    ids = torch.ones(1, device="cuda", dtype=torch.int32)
    a = torch.zeros(h, device="cuda")
    bias = torch.zeros(h, 128, device="cuda")
    gate_center = torch.linspace(-9, 9, 128, device="cuda")
    out_err, state_err = [0.0, 0.0], [0.0, 0.0]
    for step in range(64):
        if step % 16 == 0:
            checkpoint = state[1].clone()
        q, k, v, gate = [
            torch.randn(1, h, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(4)
        ]
        gate.add_(gate_center)
        beta = torch.randn(1, h, device="cuda", dtype=torch.bfloat16)
        out = cache.step(state, ids, q, k, v, gate, beta, a, bias)
        qd, kd, vd = q[0].double(), k[0].double(), v[0].double()
        qd = qd / (qd.square().sum(-1, keepdim=True) + 1e-6).sqrt() * 128**-0.5
        kd = kd / (kd.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        decay = (-5 * gate[0].double().sigmoid()).exp()
        reference *= decay[:, None, :]
        delta = beta[0].double().sigmoid()[:, None] * (
            vd - torch.einsum("hvk,hk->hv", reference, kd)
        )
        reference += delta[:, :, None] * kd[:, None, :]
        expected = torch.einsum("hvk,hk->hv", reference, qd)
        torch.testing.assert_close(out[0].double(), expected, **OUT_TOL)
        out_err = [max(x, y) for x, y in zip(out_err, _errors(out[0], expected))]
        if step % 16 == 15:
            torch.testing.assert_close(
                state[1].double(), reference, atol=3e-6, rtol=5e-5
            )
            err = _errors(state[1], reference)
            state_err = [max(x, y) for x, y in zip(state_err, err)]
        else:
            assert torch.equal(state[1], checkpoint)
    print(
        f"\nfp64: out max abs {out_err[0]:.3e} rel {out_err[1]:.3e}; "
        f"flushed state max abs {state_err[0]:.3e} rel {state_err[1]:.3e}"
    )


@torch.inference_mode()
def test_mixed_decode_never_enters_flashkda_prefill():
    """Repeated prefill arrivals must not reset an existing decode window."""
    flashkda = pytest.importorskip("vllm._flashkda_C")
    del flashkda
    torch.manual_seed(49)
    heads, tokens = 4, 66  # Two decode rows and one 64-token prefill.
    state = torch.randn(5, heads, 128, 128, device="cuda") * 0.1
    reference = state.clone()
    cache = KDAReplaySSM(heads, 4, state.device)
    solo = KDAReplaySSM(heads, 4, state.device)
    ids = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
    initial = torch.tensor([True, True, False], device="cuda")
    metadata = SimpleNamespace(
        num_decodes=2,
        num_decode_tokens=2,
        num_spec_decodes=0,
        num_prefills=1,
        num_prefill_tokens=64,
        non_spec_state_indices_tensor=ids,
        prefill_state_indices=ids[2:],
        prefill_has_initial_state=initial[2:],
        prefill_query_start_loc=torch.tensor([0, 64], device="cuda", dtype=torch.int32),
    )
    a = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads * 128, device="cuda")
    workspace = torch.empty(
        torch.ops._flashkda_C.get_workspace_size(64, heads, 1),
        device="cuda",
        dtype=torch.uint8,
    )
    last = torch.empty(1, heads, 128, 128, device="cuda")
    calls = []

    def prefill(q, k, v, g, beta, initial_state, cu_seqlens, out):
        assert q.shape[1] == 64
        calls.append(q.shape[1])
        torch.ops._flashkda_C.fwd(
            q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta,
            128**-0.5, out, workspace, a, bias.view(heads, 128), -5.0,
            initial_state.contiguous(), last, cu_seqlens.contiguous(), None, None,
        )  # fmt: skip
        return out, last

    for t in range(35):
        data = [
            torch.randn(1, tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(4)
        ]
        data[3].sub_(4)
        data.append(torch.randn(1, tokens, heads, device="cuda", dtype=torch.bfloat16))
        out = torch.empty_like(data[2])
        cache.forward(state, metadata, *data, a, bias, out, prefill)
        expected = solo.step(reference, ids[:2], *[x[0, :2] for x in data], a, bias)
        torch.testing.assert_close(out[0, :2], expected, rtol=0, atol=0)
        assert cache.pos[cache.owners > 0].tolist() == [(t + 1) % 16] * 2
        # Prefill output/state must also match a standalone native invocation.
        pf_out = torch.empty_like(out[:, 2:])
        zero = torch.zeros_like(last)
        prefill(*[x[:, 2:] for x in data], zero, metadata.prefill_query_start_loc,
                pf_out)  # fmt: skip
        torch.testing.assert_close(out[:, 2:], pf_out, rtol=0, atol=0)
        torch.testing.assert_close(state[3], last[0], rtol=0, atol=0)
    assert len(calls) == 70


@torch.inference_mode()
def test_replay_4096_step_graph_lifecycle_keeps_native_state():
    """64 slots, 96 physical pages, churn/pauses/padding and partial handoffs."""
    torch.manual_seed(300)
    h, batch = 2, 32
    state = torch.randn(97, h, 128, 128, device="cuda") * 0.1
    reference = state.clone()
    cache = KDAReplaySSM(h, 64, state.device)
    ids = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    data = _inputs(batch, h)
    a, bias = torch.zeros(h, device="cuda"), torch.zeros(h, 128, device="cuda")
    cache.step(state, ids, *data, a, bias)
    cache.reset()
    state.copy_(reference)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = cache.step(state, ids, *data, a, bias)
    cache.reset()
    state.copy_(reference)
    initial = torch.ones(1, device="cuda", dtype=torch.bool)
    out_err = [0.0, 0.0]
    for step in range(4096):
        values = [(row + 7 * (step // 37)) % 96 + 1 for row in range(batch)]
        if step % 2:
            values.reverse()
        values[-1] = 0
        ids.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
        if step % 97 == 0:
            cache.release_finished(ids[:1])
            state[values[0]].zero_()
            reference[values[0]].zero_()
        if step % 53 == 0:
            cache.before_prefill(state, ids[1:2], initial)
            torch.testing.assert_close(
                state[values[1]], reference[values[1]], rtol=1e-3, atol=4e-6
            )
        data[0].normal_()
        graph.replay()
        native = _recurrent(reference, ids, data, a, bias)
        torch.testing.assert_close(out[:-1], native[:-1], **OUT_TOL)
        out_err = [max(x, y) for x, y in zip(out_err, _errors(out[:-1], native[:-1]))]
    all_ids = torch.arange(1, 97, device="cuda", dtype=torch.int32)
    cache.before_prefill(state, all_ids, torch.ones_like(all_ids, dtype=torch.bool))
    torch.testing.assert_close(state, reference, rtol=1e-3, atol=4e-6)
    print(
        f"\n4096 steps: out max abs {out_err[0]:.3e} rel {out_err[1]:.3e}; "
        f"final state {_errors(state, reference)}"
    )


def _research_replay_cache():
    """The ReplayCache of the paper's KDA ReplaySSM baseline (``glm_window``),
    loaded from the directory in KDA_REPLAYSSM_REFERENCE_DIR."""
    root = Path(os.environ.get("KDA_REPLAYSSM_REFERENCE_DIR", "/nonexistent"))
    if not (root / "cache.py").exists():
        pytest.skip("set KDA_REPLAYSSM_REFERENCE_DIR to the reference glm_window tree")
    name = "_research_glm_window"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{name}.cache").ReplayCache


@torch.inference_mode()
def test_replay_bit_identical_to_research_kernels():
    """The port keeps the research method and kernels bit for bit."""
    ResearchCache = _research_replay_cache()
    torch.manual_seed(5)
    heads, capacity, pages = 4, 6, 11
    state = torch.randn(pages, heads, 128, 128, device="cuda") * 0.1
    research_state = state.clone()
    port = KDAReplaySSM(heads, capacity, state.device)
    research = ResearchCache(heads, capacity, state.device)
    a = 0.5 * torch.randn(heads, device="cuda")
    bias = 0.1 * torch.randn(heads, 128, device="cuda")
    for t in range(150):
        rows = torch.randperm(pages - 1)[:5] + 1
        rows[(t // 7) % 5] = 0
        ids = rows.to(device="cuda", dtype=torch.int32)
        data = _inputs(5, heads)
        if t % 13 == 0:
            port.release_finished(ids[:1])
            research.release_finished(ids[:1])
        if t % 17 == 0:
            flags = torch.tensor([True, False], device="cuda")
            port.before_prefill(state, ids[1:3], flags)
            research.before_prefill(research_state, ids[1:3], flags)
        out = port.step(state, ids, *data, a, bias)
        research_out = research.step(research_state, ids, *data, a, bias)
        assert torch.equal(out, research_out), t
        assert torch.equal(state, research_state), t
        assert torch.equal(port.owners, research.owners), t
        assert torch.equal(port.pos, research.pool.pos), t
    all_ids = torch.arange(1, pages, device="cuda", dtype=torch.int32)
    flags = torch.ones_like(all_ids, dtype=torch.bool)
    port.before_prefill(state, all_ids, flags)
    research.before_prefill(research_state, all_ids, flags)
    assert torch.equal(state, research_state)
