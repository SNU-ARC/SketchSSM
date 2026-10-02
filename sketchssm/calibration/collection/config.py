# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Validate the collection contract before starting an engine."""

from copy import deepcopy
from pathlib import Path

import yaml

from ..adapters import load_adapter
from ..core.traffic import AllocationCost

STAGES = ("generate", "covariance", "basis", "paired", "allocate")


def read_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg = deepcopy(cfg)
    if cfg.get("schema_version") != 1:
        raise ValueError("Expected schema_version: 1")
    model = cfg["model"]
    if not model.get("checkpoint"):
        raise ValueError("Set model.checkpoint to the original checkpoint or model ID")
    for key in ("checkpoint",):
        value = model[key]
        if value.startswith((".", "/")):
            resolved = Path(value) if value.startswith("/") else path.parent / value
            model[key] = str(resolved.resolve())
            if not resolved.is_dir():
                raise ValueError(f"Model directory does not exist: {resolved}")
    adapter = load_adapter(cfg.get("adapter", "auto"), family=model["family"])
    adapter.validate_geometry(cfg["geometry"])
    cfg["adapter"] = cfg.get("adapter") or model["family"]
    W = cfg["geometry"]["window"]
    gen, paired = cfg["generation"], cfg["paired"]
    if gen["temperature"] != 0 or not gen["ignore_eos"] or gen["chat_template"]:
        raise ValueError("Calibration uses greedy raw-token generation with ignore_eos")
    if (
        not gen["discard_first_window"]
        or gen["new_tokens"] % W
        or gen["new_tokens"] <= W
    ):
        raise ValueError(
            "Covariance requires full windows, discarding the first window"
        )
    if gen["split"] == paired["split"]:
        raise ValueError("Generation and paired-gradient splits must be disjoint")
    if paired["objective"] != "full-gram":
        raise ValueError("Only Full-Gram paired allocation is supported")
    if paired["warmup_tokens"] % W or paired["sequence_length"] % W:
        raise ValueError("Paired sequence length and warmup must align to the window")
    if not 0 < paired["warmup_tokens"] < paired["sequence_length"]:
        raise ValueError("Paired warmup must be positive and shorter than the sequence")
    for key in ("sequences", "prompt_tokens", "new_tokens"):
        if gen[key] < 1:
            raise ValueError(f"generation.{key} must be positive")
    if paired["sequences"] < 1 or paired["block_start"] < 0:
        raise ValueError("Invalid paired data selection")
    geo = cfg["geometry"]
    crossover = AllocationCost(
        geo["key_dim"], geo["value_dim"], W, geo["erase"]
    ).crossover
    # allocation.max_rank is optional and defaults to the dense crossover rank.
    max_rank = cfg["allocation"].get("max_rank", crossover)
    if not 1 <= max_rank <= crossover:
        raise ValueError(f"allocation.max_rank must lie in [1, {crossover}]")
    if not max_rank <= cfg["basis"]["rank"] <= geo["key_dim"]:
        raise ValueError("Require max_rank <= basis rank <= key dimension")
    if not cfg["allocation"]["mean_ranks"]:
        raise ValueError("Provide at least one mean rank")
    runtime = cfg.setdefault("runtime", {})
    runtime.setdefault("batch_size", 32)
    runtime.setdefault("device", "cuda")
    runtime.setdefault("teacher_force", "logits_processor")
    if runtime["batch_size"] < 1 or runtime["teacher_force"] not in (
        "logits_processor",
        "trace_decode",
    ):
        raise ValueError("Invalid batch size or teacher-force binding")
    loader = cfg.setdefault("gradient", {}).setdefault("loader", "ordinary")
    if (
        loader
        not in ("ordinary", "super_nvfp4", "qwen_nvfp4", "qwen3_5_nvfp4", "glm_nvfp4")
        and ":" not in loader
    ):
        raise ValueError("Unknown gradient loader; use a built-in or module:factory")
    return cfg
