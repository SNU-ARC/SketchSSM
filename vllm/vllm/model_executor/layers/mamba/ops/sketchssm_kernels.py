# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM CUDA kernels, vendored or from the ``sketchssm`` package.

Used when ``VLLM_SKETCHSSM_USE_CUDA`` is set and the layer shape is supported;
otherwise layers use the Triton kernels.
"""

import functools
import os
from types import ModuleType

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.gdn_sketchssm_common import GDNSketchArgs
from vllm.model_executor.layers.mamba.ops.kda_sketchssm_common import (
    KDA_SKETCH_LOWER_BOUND,
    KDASketchArgs,
    KDASketchRings,
)
from vllm.model_executor.layers.mamba.ops.sketchssm_mamba2 import (
    SketchArgs,
    row_list_programs,
    run_with_flush,
    sketch_bc_pre,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

logger = init_logger(__name__)

SKETCHSSM_KERNELS_API = 1


@functools.cache
def _package() -> ModuleType | None:
    if not envs.VLLM_SKETCHSSM_USE_CUDA or not current_platform.is_cuda():
        return None
    try:
        import vllm.third_party.sketchssm_kernels as kernels
    except ImportError:
        try:
            from sketchssm import kernels
        except ImportError:
            logger.info_once(
                "The SketchSSM CUDA kernels are not available; SketchSSM uses its "
                "Triton kernels"
            )
            return None
    api = getattr(kernels, "API_VERSION", None)
    if api != SKETCHSSM_KERNELS_API:
        logger.warning_once(
            "SketchSSM kernels API %s does not match vLLM's %d; SketchSSM uses "
            "its Triton kernels",
            api,
            SKETCHSSM_KERNELS_API,
        )
        return None
    kernels.set_config_dirs([envs.VLLM_TUNED_CONFIG_FOLDER])
    kernels.set_cache_dir(os.path.join(envs.VLLM_CACHE_ROOT, "sketchssm_cuda"))
    return kernels


def _kernels() -> ModuleType:
    sk = _package()
    assert sk is not None
    return sk


def _supported(family: str, support) -> bool:
    if not support:
        logger.info_once(
            "SketchSSM %s CUDA kernels unavailable (%s); using the Triton kernels",
            family,
            support.reason,
        )
    return bool(support)


def mamba2_cuda_supported(
    num_heads: int,
    head_dim: int,
    state_size: int,
    n_groups: int,
    window: int,
    activation_dtype: torch.dtype,
    state_dtype: torch.dtype,
) -> bool:
    sk = _package()
    return sk is not None and _supported(
        "Mamba-2",
        sk.mamba2_supported(
            num_heads,
            head_dim,
            state_size,
            n_groups,
            window,
            activation_dtype,
            state_dtype,
        ),  # fmt: skip
    )


def mamba2_cuda_decode(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    x_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    B_cache: torch.Tensor,
    bc_pre: torch.Tensor,
    write_pos: torch.Tensor,
    is_flush: torch.Tensor,
    flush_rows: torch.Tensor,
    slots: torch.Tensor,
    meta: torch.Tensor,
    out: torch.Tensor,
    sketch: SketchArgs,
    null_block_id: int = NULL_BLOCK_ID,
    has_flush_rows: bool = True,
) -> None:
    """Same arguments and results as ``sketch_triton_decode``."""
    batch = x.shape[0]
    if batch == 0:
        return
    if slots.dim() == 2:
        slots = slots[:, 0]
    sketch_bc_pre(B, C, B_cache, write_pos, is_flush, bc_pre, slots, null_block_id)
    _kernels().mamba2_decode(
        state, x, dt, A, B, C, D, dt_bias, x_cache, dt_cache, B_cache, bc_pre,
        write_pos, is_flush, flush_rows, slots, meta, out, sketch, null_block_id,
        has_flush_rows, flush_programs=row_list_programs(batch),
        run_with_flush=run_with_flush,
    )  # fmt: skip


def gdn_cuda_supported(
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    window: int,
    activation_dtype: torch.dtype,
    state_dtype: torch.dtype,
) -> bool:
    sk = _package()
    if sk is None or not _supported(
        "GDN",
        sk.gdn_supported(
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            window,
            activation_dtype,
            state_dtype,
        ),  # fmt: skip
    ):
        return False
    # Raises if no build knobs fit this shape and window on this GPU.
    sk.gdn.check_resources(num_k_heads, num_v_heads, window)
    return True


def gdn_cuda_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    out: torch.Tensor,
    state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    slots: torch.Tensor,
    write_pos: torch.Tensor,
    meta: torch.Tensor,
    flush_rows: torch.Tensor,
    sketch: GDNSketchArgs,
    scale: float,
    null_block_id: int = NULL_BLOCK_ID,
    has_flush_rows: bool = True,
) -> None:
    """Same arguments and results as ``gdn_sketch_triton_decode``."""
    _kernels().gdn_decode(
        mixed_qkv, a, b, A_log, dt_bias, out, state, d_cache, k_cache, g_cache,
        slots, write_pos, meta, flush_rows, sketch, scale, null_block_id,
        has_flush_rows,
    )  # fmt: skip


def kda_cuda_supported(
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    window: int,
    activation_dtype: torch.dtype,
    state_dtype: torch.dtype,
    lower_bound: float = KDA_SKETCH_LOWER_BOUND,
) -> bool:
    sk = _package()
    return sk is not None and _supported(
        "KDA",
        sk.kda_supported(
            num_heads,
            head_k_dim,
            head_v_dim,
            window,
            activation_dtype,
            state_dtype,
            lower_bound,
        ),  # fmt: skip
    )


def kda_cuda_cold_build(
    state: torch.Tensor,
    rings: KDASketchRings,
    slots: torch.Tensor,
    meta: torch.Tensor,
    rows: torch.Tensor,
    sketch: KDASketchArgs,
    scratch: torch.Tensor,
    null_block_id: int = NULL_BLOCK_ID,
) -> None:
    """Same arguments and results as ``kda_sketch_triton_cold_build``."""
    _kernels().kda_cold_build(
        state, rings, slots, meta, rows, sketch, scratch, null_block_id
    )


def kda_cuda_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    out: torch.Tensor,
    state: torch.Tensor,
    rings: KDASketchRings,
    slots: torch.Tensor,
    meta: torch.Tensor,
    pos: torch.Tensor,
    flush_rows: torch.Tensor,
    sketch: KDASketchArgs,
    scratch: torch.Tensor,
    scale: float = 128**-0.5,
    null_block_id: int = NULL_BLOCK_ID,
    has_flush_rows: bool = True,
) -> None:
    """Same arguments and results as ``kda_sketch_triton_decode``."""
    _kernels().kda_decode(
        q, k, v, g, beta, A_log, dt_bias, out, state, rings, slots, meta, pos,
        flush_rows, sketch, scratch, scale, null_block_id, has_flush_rows,
    )  # fmt: skip
