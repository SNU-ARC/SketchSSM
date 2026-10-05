# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Kimi Delta Attention decode latency per layer: standard vs SketchSSM.

Each mode runs one decode step of one KDA layer (64 heads of dim 128) for a
batch, captured in a CUDA graph: ``standard`` (vLLM's fused recurrent KDA),
``sketch`` (these CUDA kernels) and ``triton`` (vLLM's SketchSSM Triton
kernels). ``--phase`` sets every row's window position: ``nonflush``,
``flush`` or ``mixed``. SketchSSM uses the per-head ranks and frames of one
calibration layer.

Example:
    python -m sketchssm.kernels.tools.benchmark_kda \\
        --calibration glm_g7.pt --layer 10 --window 64
"""

import argparse

import torch
from vllm.model_executor.layers.mamba.ops.kda_sketchssm_common import (
    KDASketchArgs,
    KDASketchRings,
    KDASketchTables,
    kda_sketch_paged_views,
    kda_sketch_ring_specs,
    kda_sketch_scratch,
)
from vllm.model_executor.layers.mamba.ops.kda_sketchssm_triton import (
    kda_sketch_triton_cold_build,
    kda_sketch_triton_decode,
)
from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda

from sketchssm.kernels import kda as skk
from sketchssm.kernels.tools.benchmark_utils import (
    add_benchmark_args,
    flush_row_list,
    make_layout,
    ring_phase,
    run_benchmark,
)

# Heads, value and key head dims.
H, V, K = 64, 128, 128
MODES = ("standard", "sketch")


def kda_step(mode, batch, args, layout):
    dev = "cuda"
    slots = torch.arange(1, batch + 1, dtype=torch.int32, device=dev)
    state = torch.randn(batch + 1, H, V, K, device=dev) * 0.1
    q, k, v, g = (
        torch.randn(batch, H * K, device=dev, dtype=torch.bfloat16) for _ in range(4)
    )
    beta = torch.randn(batch, H, device=dev, dtype=torch.bfloat16)
    A_log = torch.zeros(H, device=dev)
    dt_bias = torch.zeros(H * K, device=dev)
    out = torch.empty(batch, H, V, device=dev, dtype=torch.bfloat16)
    if mode == "standard":
        cu = torch.arange(batch + 1, dtype=torch.int32, device=dev)
        return lambda: fused_recurrent_kda(
            q=q.view(1, batch, H, K), k=k.view(1, batch, H, K),
            v=v.view(1, batch, H, V), g=g.view(1, batch, H, K),
            beta=beta.view(1, batch, H), initial_state=state,
            use_qk_l2norm_in_kernel=True, cu_seqlens=cu, ssm_state_indices=slots,
            out=out.view(1, batch, H, V), sigmoid_beta=True,
            a_log=A_log.view(1, 1, H, 1), g_bias=dt_bias, compute_gate=True,
            lower_bound=-5.0,
        )  # fmt: skip
    write_pos, is_flush = ring_phase(args, batch, dev)
    w = args.window
    if mode == "sketch":
        build, decode = skk.kda_cold_build, skk.kda_decode
    else:
        build, decode = kda_sketch_triton_cold_build, kda_sketch_triton_decode
    tables = KDASketchTables(layout.frames, layout.ranks, w)
    sketch = KDASketchArgs.allocate(tables, batch, dev)
    specs = list(kda_sketch_ring_specs(H, w).values())
    rings = KDASketchRings(*kda_sketch_paged_views(specs, batch + 1, device=dev))
    scratch = kda_sketch_scratch(batch, tables, dev)
    meta = torch.arange(batch, dtype=torch.int32, device=dev)
    build(state, rings, slots, meta, meta, sketch, scratch)
    flush_rows = flush_row_list(is_flush, count_pad=True)
    has_flush_rows = bool(is_flush.any())
    return lambda: decode(
        q.view(batch, H, K), k.view(batch, H, K), v.view(batch, H, V),
        g.view(batch, H, K), beta, A_log, dt_bias, out, state, rings, slots,
        meta, write_pos, flush_rows, sketch, scratch,
        has_flush_rows=has_flush_rows,
    )  # fmt: skip


def kda_layout(args):
    return make_layout(args, H, H, K)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_benchmark_args(p)
    p.add_argument("--modes", nargs="+", default=MODES,
                   choices=("standard", "sketch", "triton"))  # fmt: skip
    args = p.parse_args()
    torch.set_default_device("cuda")
    run_benchmark(args, "kda", kda_layout(args), args.modes, kda_step)


if __name__ == "__main__":
    main()
