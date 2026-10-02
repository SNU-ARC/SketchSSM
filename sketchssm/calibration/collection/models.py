# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Load original checkpoints for input-gradient estimation, never a replacement model."""

import importlib
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class GradientModel:
    model: object
    logits: object
    audit: dict


def load_model(config):
    from transformers import AutoConfig, AutoModelForCausalLM

    spec, device = config["model"], config["runtime"]["device"]
    checkpoint = spec["checkpoint"]
    cfg = AutoConfig.from_pretrained(
        checkpoint, revision=spec.get("revision"), trust_remote_code=False
    )
    quant = getattr(cfg, "quantization_config", None) or getattr(
        getattr(cfg, "text_config", None), "quantization_config", None
    )
    loader = config["gradient"]["loader"]
    if ":" in loader:
        module, name = loader.split(":", 1)
        return getattr(importlib.import_module(module), name)(config)
    if loader == "ordinary":
        if quant:
            raise ValueError(
                "Quantized checkpoint requires an explicit supported packed gradient loader"
            )
        dtype = getattr(torch, config["gradient"].get("dtype", "bfloat16"))
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Ordinary gradient loader expects a floating-point dtype")
        factory = AutoModelForCausalLM
        if cfg.model_type in ("qwen3_5", "qwen3_5_moe"):
            from transformers import AutoModelForImageTextToText

            factory = AutoModelForImageTextToText
        model = factory.from_pretrained(
            checkpoint,
            revision=spec.get("revision"),
            torch_dtype=dtype,
            trust_remote_code=False,
            attn_implementation="eager",
        ).to(device)
        audit = dict(packed=False, temporary_weight_reconstruction=False)
    else:
        if not quant:
            raise ValueError(
                "Packed gradient loader requires a quantized source checkpoint"
            )
        if not Path(checkpoint).is_dir():
            from huggingface_hub import snapshot_download

            checkpoint = snapshot_download(checkpoint, revision=spec.get("revision"))
        name = {
            "super_nvfp4": "super",
            "qwen_nvfp4": "qwen",
            "qwen3_5_nvfp4": "qwen3_5",
            "glm_nvfp4": "glm",
        }[loader]
        module = importlib.import_module(f"{__package__}.packed.{name}")
        result = module.load(checkpoint, device=device, purpose="paired_gradient")
        if name == "glm":
            model, head, audit = result
            return GradientModel(
                model, lambda ids: module.forward(model, head, ids), audit
            )
        model = result
        audit = dict(
            packed=True, temporary_weight_reconstruction=True, native_backward=False
        )
    model.eval().requires_grad_(False)
    model.enable_input_require_grads()
    return GradientModel(
        model,
        lambda ids: model(input_ids=ids.to(device), use_cache=False).logits,
        audit,
    )
