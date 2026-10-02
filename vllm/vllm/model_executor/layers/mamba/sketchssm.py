# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loading of SketchSSM calibrations (``--sketchssm``)."""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba import sketchssm_calibration

logger = init_logger(__name__)

SKETCHSSM_CALIBRATION_FILE = "calibration.pt"
DEFAULT_SKETCHSSM_MEAN_RANK = 8.0


@dataclass(frozen=True)
class SketchSSMCalibration:
    frames: torch.Tensor  # (num_layers, ngroups, K, K) float32
    ranks: torch.Tensor  # (num_layers, nheads) int32
    layer_ids: tuple[int, ...] | None

    @property
    def num_layers(self) -> int:
        return self.ranks.shape[0]

    def layer_index(self, recurrent_idx: int, model_layer_idx: int) -> int:
        """Calibration row of a recurrent layer, by model layer id if recorded."""
        if self.layer_ids is not None:
            if model_layer_idx not in self.layer_ids:
                raise ValueError(
                    f"SketchSSM calibration has no entry for layer {model_layer_idx}"
                )
            return self.layer_ids.index(model_layer_idx)
        if not 0 <= recurrent_idx < self.num_layers:
            raise ValueError(
                f"SketchSSM calibration covers {self.num_layers} recurrent layers, "
                f"got layer {recurrent_idx}"
            )
        return recurrent_idx


def resolve_sketchssm_path(path_or_repo: str) -> str:
    """Local path of ``--sketchssm``: a file, a directory or an HF repo id."""
    path = Path(path_or_repo).expanduser()
    if path.is_dir():
        path = path / SKETCHSSM_CALIBRATION_FILE
    if path.is_file():
        return str(path)
    if path_or_repo.endswith(".pt") or path_or_repo.startswith(("/", ".", "~")):
        raise FileNotFoundError(f"SketchSSM calibration {str(path)!r} not found")
    from vllm.transformers_utils.repo_utils import hf_api

    try:
        return hf_api().hf_hub_download(path_or_repo, SKETCHSSM_CALIBRATION_FILE)
    except Exception as e:
        raise ValueError(
            f"--sketchssm {path_or_repo!r} is neither a local file nor a "
            f"Hugging Face repo with {SKETCHSSM_CALIBRATION_FILE}: {e}"
        ) from e


@cache
def load_sketchssm_calibration(
    path_or_repo: str, mean_rank: float | None = None, window: int | None = None
) -> SketchSSMCalibration:
    """Load and validate the ``--sketchssm`` calibration (cached)."""
    path = resolve_sketchssm_path(path_or_repo)
    data = torch.load(path, map_location="cpu", weights_only=True)
    if sketchssm_calibration.is_portable_calibration(data):
        if mean_rank is None:
            mean_rank = DEFAULT_SKETCHSSM_MEAN_RANK
        data = sketchssm_calibration.frames_for_mean_rank(data, mean_rank, window)
        ranks = data["m_table"]
        if data["unverified_window"]:
            logger.warning(
                "SketchSSM calibration %s: mean rank %s is unverified at window "
                "%d; its verified table applies to the calibrated window only",
                path_or_repo,
                sketchssm_calibration.rank_key(mean_rank),
                data["window"],
            )
        logger.info(
            "SketchSSM calibration %s: mean rank %s at window %d (%s), %d sketch "
            "and %d dense heads over %d layers",
            path_or_repo,
            sketchssm_calibration.rank_key(mean_rank),
            data["window"],
            "verified" if data["verified"] else "not a verified rank",
            int((ranks > 0).sum()),
            int((ranks == 0).sum()),
            ranks.shape[0],
        )
    elif mean_rank is not None:
        raise ValueError(
            f"SketchSSM calibration {path_or_repo!r} holds exported frames whose "
            "ranks are already fixed; --sketchssm-mean-rank applies only to a "
            "portable calibration file"
        )
    frames = data["frames"].to(torch.float32)
    ranks = data["m_table"]
    if frames.dim() != 4 or frames.shape[-1] != frames.shape[-2]:
        raise ValueError("SketchSSM frames must be (layers, groups, K, K)")
    if ranks.dim() != 2 or ranks.shape[0] != frames.shape[0]:
        raise ValueError("SketchSSM m_table must be (layers, heads)")
    if ranks.shape[1] % frames.shape[1] != 0:
        raise ValueError("SketchSSM heads must divide evenly into groups")
    if (ranks < 0).any() or (ranks > frames.shape[-1]).any():
        raise ValueError("SketchSSM ranks must lie in [0, K]")
    dense = data.get("dense_table")
    if dense is not None and not torch.equal(dense.bool(), ranks == 0):
        raise ValueError("SketchSSM dense_table must mark exactly the rank-0 heads")
    eye = torch.eye(frames.shape[-1], dtype=torch.float64, device="cpu")
    frames64 = frames.double()
    ortho_err = (frames64 @ frames64.transpose(-1, -2) - eye).abs().amax()
    if ortho_err > 1e-4:
        raise ValueError(f"SketchSSM frames are not orthogonal (error {ortho_err})")
    layer_ids = data.get("layer_ids")
    return SketchSSMCalibration(
        frames=frames.contiguous(),
        ranks=ranks.to(torch.int32).contiguous(),
        layer_ids=tuple(int(i) for i in layer_ids) if layer_ids is not None else None,
    )
