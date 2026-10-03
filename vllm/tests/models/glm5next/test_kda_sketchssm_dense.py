# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA SketchSSM dense heads across prefill and decode.

Dense heads read BF16 state rows built after prefill and at each flush; their
FP32 state changes only at flushes and must then match the per-step
recurrence. Covers chunked prefill and mixed prefill + decode steps.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba import kda_sketchssm as kds
from vllm.model_executor.layers.mamba.kda_sketchssm import KDASketchSSM
from vllm.model_executor.layers.mamba.ops import kda_sketchssm_common as common
from vllm.model_executor.layers.mamba.ops import kda_sketchssm_triton as kdt
from vllm.model_executor.layers.mamba.ops import sketchssm_kernels as skk
from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA")

D, W = 128, 16
RANKS = [0, 9, 128, 30, 0, 70]
H = len(RANKS)
DENSE = [0, 2, 4]
STATE_TOL = dict(rtol=3e-5, atol=2e-6)
SKETCH_STATE_TOL = dict(rtol=3e-5, atol=2e-5)  # the sketch heads' WY fold
CUDA = skk.kda_cuda_supported(H, D, D, W, torch.bfloat16, torch.float32)
BACKENDS = [
    pytest.param("cuda", marks=pytest.mark.skipif(not CUDA, reason="sketchssm")),
    "triton",
]
i32 = lambda x: torch.tensor(x, device="cuda", dtype=torch.int32)


def _recurrent(state, ids, data, a, bias):
    """Exact one-token-per-row recurrence; updates ``state`` pages."""
    cu = torch.arange(ids.numel() + 1, device="cuda", dtype=torch.int32)
    out, _ = fused_recurrent_kda(
        q=data[0].unsqueeze(0), k=data[1].unsqueeze(0), v=data[2].unsqueeze(0),
        g=data[3].unsqueeze(0), beta=data[4].unsqueeze(0), a_log=a, g_bias=bias,
        initial_state=state, cu_seqlens=cu, ssm_state_indices=ids,
        use_qk_l2norm_in_kernel=True, sigmoid_beta=True, compute_gate=True,
        lower_bound=-5.0,
    )  # fmt: skip
    return out[0]


def _inputs(n):
    data = [torch.randn(n, H, D, device="cuda").bfloat16() for _ in range(4)]
    data[3] -= 3.0
    return data + [torch.randn(n, H, device="cuda").bfloat16()]


@pytest.mark.parametrize("backend", BACKENDS)
@torch.inference_mode()
def test_dense_heads_prefill_then_decode(backend, monkeypatch):
    torch.manual_seed(5)
    if backend == "triton":
        monkeypatch.setattr(kds, "kda_cuda_supported", lambda *a, **k: False)
    frames = torch.linalg.qr(torch.randn(H, D, D, dtype=torch.float64)).Q.float()
    cal = SimpleNamespace(frames=[frames.mT.contiguous()], ranks=[torch.tensor(RANKS)])
    with torch.device("cuda"):
        layer = KDASketchSSM(cal, 0, H, D, 0, 1, W, 4, torch.bfloat16,
                             torch.float32, -5.0)  # fmt: skip
    assert (layer._decode is kdt.kda_sketch_triton_decode) == (backend == "triton")
    specs = [((H, D, D), torch.float32), *common.kda_sketch_ring_specs(H, W).values()]
    page = (sum(math.prod(s) * t.itemsize for s, t in specs) + 511) // 512 * 512
    views = common.kda_sketch_paged_views(specs, 5, page, "cuda")
    state, rings = views[0], views[1:]
    expected = torch.zeros_like(state)
    a = 0.3 * torch.randn(H, device="cuda")
    bias = 0.1 * torch.randn(H * D, device="cuda")
    req = {1: 0, 2: 1, 3: 2}
    pos: dict[int, int] = {}

    def prefill(pages, lens, done):
        for p, n in zip(pages, lens):  # the prompt tokens, one at a time
            for _ in range(n):
                data = _inputs(1)
                _recurrent(state, i32([p]), data, a, bias.view(H, D))
                _recurrent(expected, i32([p]), data, a, bias.view(H, D))
        rows = [i for i, d in enumerate(done) if d]
        meta = SimpleNamespace(
            sketch_build_rows_p=i32(rows + [-1] * (len(pages) - len(rows))),
            sketch_meta_p=i32([req[p] for p in pages]),
        )
        layer.prefilled(state, rings, meta, i32(pages))
        for p in (p for p, d in zip(pages, done) if d):
            pos[p] = 0
            got = layer.sketch.dense[req[p]]
            assert torch.equal(got, state[p, DENSE].bfloat16())

    def decode(pages):
        data = _inputs(len(pages))
        ps = [pos[p] for p in pages]
        flush = [i for i, x in enumerate(ps) if x == W - 1]
        meta = SimpleNamespace(
            sketch_meta_d=i32([req[p] for p in pages]),
            sketchssm_window_pos_d=i32(ps),
            sketch_flush_rows_d=i32(flush + [-1] * (len(pages) - len(flush))),
            sketch_has_flush_rows=bool(flush),
        )
        before = state.clone()
        out = torch.empty(len(pages), H, D, device="cuda", dtype=torch.bfloat16)
        q, k, v, g, beta = data
        layer.decode(q.flatten(1), k.flatten(1), v.flatten(1), g.flatten(1), beta,
                     a, bias, out, state, rings, meta, i32(pages))  # fmt: skip
        ref = _recurrent(expected, i32(pages), data, a, bias.view(H, D))
        dense = out[:, DENSE].double()
        want = ref[:, DENSE].double()
        assert (dense - want).norm() / want.norm() < 8e-3
        for i, p in enumerate(pages):
            if ps[i] == W - 1:
                torch.testing.assert_close(
                    state[p, DENSE], expected[p, DENSE], **STATE_TOL
                )
                torch.testing.assert_close(state[p], expected[p], **SKETCH_STATE_TOL)
                torch.testing.assert_close(out[i], ref[i], rtol=0.01, atol=2e-4)
                rows = layer.sketch.dense[req[p]]
                assert torch.equal(rows, state[p, DENSE].bfloat16())
            else:
                assert torch.equal(state[p, DENSE], before[p, DENSE])
            pos[p] = (ps[i] + 1) % W

    prefill([1], [37], [True])
    for _ in range(5):
        decode([1])
    # Mixed steps: decode rows first, then a chunked prefill of page 2.
    decode([1])
    prefill([2], [20], [False])
    decode([1])
    prefill([2], [9], [True])
    for t in range(40):
        decode([2, 1] if t % 2 else [1, 2])
        if t == 17:
            prefill([3], [25], [True])
    for _ in range(20):
        decode([3, 1, 2])
