# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM of Kimi Delta Attention layers."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.mamba.ops.kda_sketchssm_common import (
    KDA_SKETCH_HEAD_DIM,
    KDA_SKETCH_LOWER_BOUND,
    KDASketchArgs,
    KDASketchRings,
    KDASketchTables,
    kda_sketch_scratch_numel,
    kda_sketch_window_supported,
)
from vllm.model_executor.layers.mamba.ops.kda_sketchssm_triton import (
    kda_sketch_triton_cold_build,
    kda_sketch_triton_decode,
)
from vllm.model_executor.layers.mamba.ops.sketchssm_kernels import (
    kda_cuda_cold_build,
    kda_cuda_decode,
    kda_cuda_supported,
)
from vllm.model_executor.layers.mamba.sketchssm import (
    SketchSSMCalibration,
    load_sketchssm_calibration,
)

if TYPE_CHECKING:
    from vllm.config import CacheConfig


_kda_scratch: dict[str, torch.Tensor] = {}


def _kda_shared_scratch(device: torch.device, numel: int) -> torch.Tensor:
    """FP32 flush scratch shared by every KDA layer on ``device``."""
    key = str(device)
    scratch = _kda_scratch.get(key)
    if scratch is None or scratch.numel() < numel:
        scratch = torch.empty(numel, dtype=torch.float32, device=device)
        _kda_scratch[key] = scratch
    return scratch


class KDASketchSSM(torch.nn.Module):
    """SketchSSM decode of one Kimi Delta Attention layer."""

    def __init__(
        self,
        calibration: SketchSSMCalibration,
        layer: int,
        num_heads: int,
        head_dim: int,
        tp_rank: int,
        tp_size: int,
        window: int,
        max_num_reqs: int,
        activation_dtype: torch.dtype,
        state_dtype: torch.dtype,
        lower_bound: float,
    ):
        super().__init__()
        frames = calibration.frames[layer]
        ranks = calibration.ranks[layer]
        if frames.shape != (num_heads, head_dim, head_dim):
            raise ValueError(
                f"SketchSSM frames {tuple(frames.shape)} do not match "
                f"{num_heads} heads of dim {head_dim}"
            )
        if ranks.shape != (num_heads,):
            raise ValueError(
                f"SketchSSM ranks cover {ranks.numel()} of {num_heads} heads"
            )
        local = num_heads // tp_size
        heads = slice(tp_rank * local, (tp_rank + 1) * local)
        if (
            head_dim != KDA_SKETCH_HEAD_DIM
            or not kda_sketch_window_supported(window)
            or lower_bound != KDA_SKETCH_LOWER_BOUND
            or activation_dtype != torch.bfloat16
            or state_dtype != torch.float32
        ):
            raise NotImplementedError(
                f"SketchSSM for KDA needs head dim {KDA_SKETCH_HEAD_DIM}, a window "
                f"that is a multiple of 16 (got {window}), gate lower bound "
                f"{KDA_SKETCH_LOWER_BOUND}, BF16 activations and an FP32 state"
            )
        use_cuda = kda_cuda_supported(
            local, head_dim, head_dim, window, activation_dtype, state_dtype,
            lower_bound,
        )  # fmt: skip
        self._decode = kda_cuda_decode if use_cuda else kda_sketch_triton_decode
        self._cold_build = (
            kda_cuda_cold_build if use_cuda else kda_sketch_triton_cold_build
        )
        device = torch.get_default_device()
        self.tables = KDASketchTables(frames[heads].to(device), ranks[heads], window)
        self.sketch = KDASketchArgs.allocate(self.tables, max_num_reqs, device)
        # Flushes and cold builds cover at most one row per request.
        self._scratch_numel = kda_sketch_scratch_numel(
            max_num_reqs, self.tables.num_sketch_heads
        )
        self._device = device
        _kda_shared_scratch(device, self._scratch_numel)

    def _scratch(self) -> torch.Tensor:
        return _kda_shared_scratch(self._device, self._scratch_numel)

    def prefilled(
        self,
        state: torch.Tensor,
        rings: Sequence[torch.Tensor],
        attn_metadata,
        state_indices: torch.Tensor,
    ) -> None:
        """Build the sketch of each prefill row that completes its prompt."""
        rows = attn_metadata.sketch_build_rows_p
        n = rows.numel()
        if n == 0:
            return
        self._cold_build(
            state, KDASketchRings(*rings), state_indices[:n].contiguous(),
            attn_metadata.sketch_meta_p, rows, self.sketch, self._scratch(),
        )  # fmt: skip

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        out: torch.Tensor,
        state: torch.Tensor,
        rings: Sequence[torch.Tensor],
        attn_metadata,
        state_indices: torch.Tensor,
    ) -> None:
        """One decode step on pre-activation, possibly row-strided q/k/v/g/beta."""
        n, h = q.shape[0], self.tables.num_heads
        if state_indices.dim() == 2:
            state_indices = state_indices[:, 0]
        self._decode(
            q.view(n, h, -1), k.view(n, h, -1), v.view(n, h, -1),
            g.view(n, h, -1), beta, A_log, dt_bias, out, state,
            KDASketchRings(*rings), state_indices.contiguous(),
            attn_metadata.sketch_meta_d, attn_metadata.sketchssm_window_pos_d,
            attn_metadata.sketch_flush_rows_d, self.sketch, self._scratch(),
            has_flush_rows=attn_metadata.sketch_has_flush_rows,
        )  # fmt: skip

    @classmethod
    def maybe_create(
        cls,
        cache_config: "CacheConfig | None",
        recurrent_layer_idx: int,
        model_layer_idx: int,
        num_heads: int,
        head_dim: int,
        tp_rank: int,
        tp_size: int,
        max_num_reqs: int,
        activation_dtype: torch.dtype,
        state_dtype: torch.dtype,
        lower_bound: float,
    ) -> "KDASketchSSM | None":
        """The layer's SketchSSM when ``--sketchssm`` is set."""
        if cache_config is None or cache_config.sketchssm is None:
            return None
        calibration = load_sketchssm_calibration(
            cache_config.sketchssm,
            cache_config.sketchssm_mean_rank,
            cache_config.replayssm_buffer_len,
        )
        return cls(
            calibration,
            calibration.layer_index(recurrent_layer_idx, model_layer_idx),
            num_heads,
            head_dim,
            tp_rank,
            tp_size,
            cache_config.replayssm_buffer_len,
            max_num_reqs,
            activation_dtype,
            state_dtype,
            lower_bound,
        )
