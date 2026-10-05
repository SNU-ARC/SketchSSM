# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Gated DeltaNet decode latency per layer: standard vs SketchSSM.

Each mode runs one decode step of one GDN layer for a batch, captured in a
CUDA graph: ``standard`` (vLLM's fused recurrent decode), ``sketch`` (these
CUDA kernels) and ``triton`` (vLLM's SketchSSM Triton kernels). ``--phase``
sets every row's window position: ``nonflush``, ``flush`` or ``mixed``.
SketchSSM uses the per-head ranks of one calibration layer; for a head count
the calibration does not cover, the ranks are cycled over the heads, with
random orthogonal frames.

Example:
    python -m sketchssm.kernels.tools.benchmark_gdn \\
        --calibration qwen_g7.pt --layer 10
"""

import argparse

import torch
from vllm.model_executor.layers.mamba.ops.gdn_sketchssm_common import (
    GDNSketchArgs,
    GDNSketchTables,
    gdn_rotation_from_frames,
    gdn_sketch_build,
    gdn_sketch_rotate_,
)
from vllm.model_executor.layers.mamba.ops.gdn_sketchssm_triton import (
    gdn_sketch_triton_decode,
)
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
)

from sketchssm.kernels import gdn as skg
from sketchssm.kernels.tools.benchmark_utils import (
    add_benchmark_args,
    flush_row_list,
    make_layout,
    ring_phase,
    run_benchmark,
)

# Value and key head dims.
V = K = 128
MODES = ("standard", "sketch")


def gdn_step(mode, batch, args, layout):
    HV, H, w, dev = args.num_v_heads, args.num_k_heads, args.window, "cuda"
    slots = torch.arange(1, batch + 1, dtype=torch.int32, device=dev)
    state = torch.randn(batch + 1, HV, V, K, device=dev) * 0.1
    mixed = torch.randn(batch, 2 * H * K + HV * V, device=dev, dtype=torch.bfloat16)
    a = torch.randn(batch, HV, device=dev, dtype=torch.bfloat16)
    b = torch.randn(batch, HV, device=dev, dtype=torch.bfloat16)
    A_log = torch.zeros(HV, device=dev)
    dt_bias = torch.zeros(HV, device=dev)
    out = torch.empty(batch, 1, HV, V, device=dev, dtype=torch.bfloat16)
    common = dict(mixed_qkv=mixed, a=a, b=b, A_log=A_log, dt_bias=dt_bias)
    if mode == "standard":
        return lambda: fused_recurrent_gated_delta_rule_packed_decode(
            **common, scale=K**-0.5, initial_state=state, out=out,
            ssm_state_indices=slots, use_qk_l2norm_in_kernel=True,
        )  # fmt: skip
    write_pos, is_flush = ring_phase(args, batch, dev)
    d = torch.zeros(batch + 1, HV, w, V, device=dev, dtype=torch.bfloat16)
    k = torch.zeros(batch + 1, H, w, K, device=dev, dtype=torch.bfloat16)
    g = torch.zeros(batch + 1, HV, w, device=dev)
    tables = GDNSketchTables(layout.ranks, H, w)
    sketch = GDNSketchArgs.allocate(tables, batch)
    meta = torch.arange(batch, dtype=torch.int32)
    rotation_t = gdn_rotation_from_frames(layout.frames)
    decode = skg.gdn_decode if mode == "sketch" else gdn_sketch_triton_decode
    gdn_sketch_build(state, torch.ones_like(slots), slots, meta, sketch)
    flush_rows = flush_row_list(is_flush, count_pad=True)
    has_flush_rows = bool(is_flush.any())

    def step():
        gdn_sketch_rotate_(mixed, rotation_t)
        decode(
            **common, out=out.view(batch, HV, V), state=state, d_cache=d,
            k_cache=k, g_cache=g, slots=slots, write_pos=write_pos, meta=meta,
            flush_rows=flush_rows, sketch=sketch, scale=K**-0.5,
            has_flush_rows=has_flush_rows,
        )  # fmt: skip

    return step


def add_shape_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--num-k-heads", type=int, default=16, help="key heads")
    p.add_argument("--num-v-heads", type=int, default=48, help="value heads")


def gdn_layout(args):
    return make_layout(args, args.num_v_heads, args.num_k_heads, K)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_benchmark_args(p)
    add_shape_args(p)
    p.add_argument("--modes", nargs="+", default=MODES,
                   choices=("standard", "sketch", "triton"))  # fmt: skip
    args = p.parse_args()
    torch.set_default_device("cuda")
    run_benchmark(args, "gdn", gdn_layout(args), args.modes, gdn_step)


if __name__ == "__main__":
    main()
