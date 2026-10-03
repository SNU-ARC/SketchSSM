# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM of Gated DeltaNet layers.

Dense heads (rank 0) of a calibration keep BF16 rows of their full state,
read on non-flush steps. ``--use-replayssm`` runs the same kernels with every
head dense, without those rows (the FP32 state is read at every step) and no
rotation: the exact window replay of ReplaySSM, without a calibration.
"""

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.mamba.ops.gdn_sketchssm_common import (
    GDN_SKETCH_HEAD_DIM,
    GDN_SKETCH_WINDOW_ALIGN,
    GDNSketchArgs,
    GDNSketchTables,
    gdn_rotation_from_frames,
    gdn_sketch_build,
    gdn_sketch_rotate_,
    gdn_sketch_window_supported,
)
from vllm.model_executor.layers.mamba.ops.gdn_sketchssm_triton import (
    gdn_sketch_triton_decode,
)
from vllm.model_executor.layers.mamba.ops.sketchssm_kernels import (
    gdn_cuda_decode,
    gdn_cuda_supported,
)
from vllm.model_executor.layers.mamba.sketchssm import load_sketchssm_calibration

if TYPE_CHECKING:
    from vllm.config import CacheConfig


class GDNSketchSSM(torch.nn.Module):
    """SketchSSM frames, sketch and decode kernels of one GDN layer.

    ``frames`` None keeps q and k unrotated and dense heads without BF16
    rows (ReplaySSM, all ranks 0).
    """

    def __init__(
        self,
        ranks: torch.Tensor,
        frames: torch.Tensor | None,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        window: int,
        max_num_reqs: int,
        activation_dtype: torch.dtype,
        state_dtype: torch.dtype,
    ):
        super().__init__()
        frames_shape = (num_k_heads, head_k_dim, head_k_dim)
        if frames is not None and frames.shape != frames_shape:
            raise ValueError(
                f"SketchSSM frames {tuple(frames.shape)} do not match "
                f"{num_k_heads} key heads of dim {head_k_dim}"
            )
        if ranks.shape != (num_v_heads,):
            raise ValueError(
                f"SketchSSM ranks cover {ranks.numel()} of {num_v_heads} heads"
            )
        if (
            head_k_dim != GDN_SKETCH_HEAD_DIM
            or head_v_dim != GDN_SKETCH_HEAD_DIM
            or not gdn_sketch_window_supported(window)
            or num_v_heads % num_k_heads
            or state_dtype != torch.float32
        ):
            raise NotImplementedError(
                "SketchSSM for Gated DeltaNet needs head dims "
                f"{GDN_SKETCH_HEAD_DIM}, a window that is a multiple of "
                f"{GDN_SKETCH_WINDOW_ALIGN}, a whole number of value heads per key "
                "head and an FP32 state"
            )
        use_cuda = gdn_cuda_supported(
            num_k_heads, num_v_heads, head_k_dim, head_v_dim, window,
            activation_dtype, state_dtype,
        )  # fmt: skip
        self._decode = gdn_cuda_decode if use_cuda else gdn_sketch_triton_decode
        device = torch.get_default_device()
        rotation_t = None
        if frames is not None:
            rotation_t = gdn_rotation_from_frames(frames).to(device)
        self.register_buffer("rotation_t", rotation_t, persistent=False)
        self.tables = GDNSketchTables(
            ranks, num_k_heads, window, dense_rows=frames is not None
        )
        self.sketch = GDNSketchArgs.allocate(self.tables, max_num_reqs, device)

    def rotate_(self, mixed_qkv: torch.Tensor) -> None:
        """Rotate q and k of ``mixed_qkv (tokens, 2 H K + HV V)`` in place."""
        if self.rotation_t is not None:
            gdn_sketch_rotate_(mixed_qkv, self.rotation_t)

    def prefilled(
        self, state: torch.Tensor, attn_metadata, state_indices: torch.Tensor
    ) -> None:
        """Build the sketch of each prefill row that completes its prompt."""
        gdn_sketch_build(
            state, attn_metadata.sketch_build_p, state_indices,
            attn_metadata.sketch_meta_p, self.sketch,
        )  # fmt: skip

    def decode(
        self,
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
        attn_metadata,
        state_indices: torch.Tensor,
        scale: float,
    ) -> None:
        """One decode step; rows at the end of their window are flushed."""
        self._decode(
            mixed_qkv, a, b, A_log, dt_bias, out, state, d_cache, k_cache,
            g_cache, state_indices, attn_metadata.sketchssm_window_pos_d,
            attn_metadata.sketch_meta_d, attn_metadata.sketch_flush_rows_d,
            self.sketch, scale,
        )  # fmt: skip

    @classmethod
    def maybe_create(
        cls,
        cache_config: "CacheConfig | None",
        recurrent_layer_idx: int,
        model_layer_idx: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        max_num_reqs: int,
        activation_dtype: torch.dtype,
        state_dtype: torch.dtype,
    ) -> "GDNSketchSSM | None":
        """The layer's SketchSSM when ``--sketchssm`` is set, or its dense
        ReplaySSM for ``--use-replayssm``."""
        if cache_config is None or not cache_config.uses_gdn_sketchssm:
            return None
        if cache_config.sketchssm is None:
            ranks, frames = torch.zeros(num_v_heads, dtype=torch.int32), None
        else:
            calibration = load_sketchssm_calibration(
                cache_config.sketchssm,
                cache_config.sketchssm_mean_rank,
                cache_config.replayssm_buffer_len,
            )
            layer = calibration.layer_index(recurrent_layer_idx, model_layer_idx)
            ranks, frames = calibration.ranks[layer], calibration.frames[layer]
        return cls(
            ranks,
            frames,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            cache_config.replayssm_buffer_len,
            max_num_reqs,
            activation_dtype,
            state_dtype,
        )
