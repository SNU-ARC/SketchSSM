# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Mamba-2 decode latency per layer: standard vs ReplaySSM vs SketchSSM.

Each mode runs one decode step of one Mamba-2 layer for a batch, captured in
a CUDA graph: ``standard`` (vLLM's selective state update), ``replay``
(ReplaySSM), ``sketch`` (these CUDA kernels) and ``triton`` (vLLM's SketchSSM
Triton kernels). ``--phase`` sets every row's window position: ``nonflush``,
``flush`` (the window's last step) or ``mixed`` (rows spread over the window).
SketchSSM uses the per-head ranks of one calibration layer; for a shape the
calibration does not cover, the ranks are cycled over the heads and scaled to
the state size, with random orthogonal frames.

Example:
    python -m sketchssm.kernels.tools.benchmark_mamba2 \\
        --calibration nano_g6.pt --layer 13 --head-dim 80
"""

import argparse

import torch
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2 as sk_ops
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_output_only import (  # noqa: E501
    selective_state_update_replayssm_output_only,
)
from vllm.model_executor.layers.mamba.ops.sketchssm_mamba2_triton import (
    sketch_triton_decode,
)

from sketchssm.kernels import mamba2 as skm
from sketchssm.kernels.tools.benchmark_utils import (
    add_benchmark_args,
    flush_row_list,
    make_layout,
    ring_phase,
    run_benchmark,
)

MODES = ("standard", "replay", "sketch")


def cuda_decode(state, x, dt, A, B, C, D, dt_bias, x_cache, dt_cache, B_cache,
                bc_pre, write_pos, is_flush, flush_rows, slots, meta, out, sketch,
                null_block_id=0, has_flush_rows=True):  # fmt: skip
    """These kernels as vLLM runs them: ``bc_pre`` and the FP32 query from
    vLLM's Triton kernels, and the flush on a side stream."""
    sk_ops.sketch_bc_pre(B, C, B_cache, write_pos, is_flush, bc_pre, slots,
                         null_block_id)  # fmt: skip
    query = sk_ops.sketch_query(C, sketch.frames_t)
    skm.mamba2_decode(
        state, x, dt, A, B, query, D, dt_bias, x_cache, dt_cache, B_cache, bc_pre,
        write_pos, is_flush, flush_rows, slots, meta, out, sketch, null_block_id,
        has_flush_rows, frames_t=sketch.frames_t,
        flush_programs=sk_ops.row_list_programs(x.shape[0]),
        run_with_flush=sk_ops.run_with_flush,
    )  # fmt: skip


def mamba2_step(mode, batch, args, layout):
    H, G, P, N = args.num_heads, args.ngroups, args.head_dim, args.state_size
    L, dev = args.window, "cuda"
    slots = torch.arange(1, batch + 1, dtype=torch.int32, device=dev)
    state = torch.randn(batch + 1, H, P, N, device=dev) * 0.1
    # x, B and C are column slices of one conv output row, as in the model.
    xbc = torch.randn(batch, H * P + 2 * G * N, device=dev, dtype=torch.bfloat16)
    x = xbc[:, : H * P].view(batch, H, P)
    B = xbc[:, H * P : H * P + G * N].view(batch, G, N)
    C = xbc[:, H * P + G * N :].view(batch, G, N)
    dt = torch.randn(batch, H, device=dev)[:, :, None].expand(-1, -1, P)
    A = (-torch.rand(H, device=dev))[:, None, None].expand(-1, P, N)
    D = torch.rand(H, device=dev)[:, None].expand(-1, P)
    dt_bias = torch.rand(H, device=dev)[:, None].expand(-1, P)
    out = torch.empty_like(x)
    if mode == "standard":
        return lambda: selective_state_update(
            state, x, dt, A, B, C, D, dt_bias, dt_softplus=True,
            state_batch_indices=slots, dst_state_batch_indices=slots, out=out,
        )  # fmt: skip
    write_pos, is_flush = ring_phase(args, batch, dev)
    rings = dict(
        x_cache=torch.randn(batch + 1, H, L, P, device=dev, dtype=torch.bfloat16),
        dt_cache=torch.rand(batch + 1, H, L, device=dev) * 0.1,
        B_cache=torch.randn(batch + 1, G, L, N, device=dev, dtype=torch.bfloat16),
        bc_pre=torch.empty(batch, G, L, device=dev),
    )
    if mode in ("sketch", "triton"):
        decode = cuda_decode if mode == "sketch" else sketch_triton_decode
        return mamba2_sketch_step(
            decode, batch, layout, state, x, dt, A, B, C, D, dt_bias, rings,
            write_pos, is_flush, slots, out,
        )  # fmt: skip

    def step():
        selective_state_update_replayssm_output_only(
            state, x, dt, A, B, C, D, dt_bias, dt_softplus=True, **rings,
            write_pos=write_pos, is_flush=is_flush, max_cache_len=L,
            state_batch_indices=slots, out=out,
        )  # fmt: skip

    return step


def mamba2_sketch_step(decode, batch, layout, state, x, dt, A, B, C, D, dt_bias,
                       rings, write_pos, is_flush, slots, out):  # fmt: skip
    """SketchSSM as the model runs it: the layer's frames, key-major state and
    BF16 dt/D/dt_bias."""
    heads, dim, dstate = state.shape[1:]
    state = state.transpose(-1, -2).contiguous().transpose(-1, -2)
    ranks = layout.ranks.cpu()
    shapes = sk_ops.sketch_shapes(ranks, dim, dstate)
    buffers = (torch.zeros(batch, *s, dtype=d)
               for s, d in zip(shapes, sk_ops.SKETCH_DTYPES))  # fmt: skip
    sketch = sk_ops.SketchArgs(
        *buffers, tables=sk_ops.SketchTables(ranks, dstate).cuda(),
        frames_t=layout.frames_t,
    )
    meta = torch.arange(batch, dtype=torch.int32)
    dt = dt[:, :, :1].bfloat16().expand(-1, -1, dim)
    D, dt_bias = D.bfloat16(), dt_bias.bfloat16()
    sk_ops.sketch_build(state, torch.ones_like(is_flush), slots, meta, sketch)
    # A step with no flushing row skips the flush launch.
    has_flush_rows = bool(is_flush.any())
    flush_rows = flush_row_list(is_flush)

    def step():
        decode(
            state, x, dt, A, B, C, D, dt_bias, rings["x_cache"], rings["dt_cache"],
            rings["B_cache"], rings["bc_pre"], write_pos, is_flush, flush_rows,
            slots, meta, out, sketch, has_flush_rows=has_flush_rows,
        )  # fmt: skip

    return step


def add_shape_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--num-heads", type=int, default=128)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--ngroups", type=int, default=8)
    p.add_argument("--state-size", type=int, default=128)


def mamba2_layout(args):
    return make_layout(args, args.num_heads, args.ngroups, args.state_size)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_benchmark_args(p)
    add_shape_args(p)
    p.add_argument("--modes", nargs="+", default=MODES,
                   choices=("standard", "replay", "sketch", "triton"))  # fmt: skip
    args = p.parse_args()
    torch.set_default_device("cuda")
    run_benchmark(args, "mamba2", mamba2_layout(args), args.modes, mamba2_step)


if __name__ == "__main__":
    main()
