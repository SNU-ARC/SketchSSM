# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"Packed Qwen3.5-architecture loader restricted to paired gradient estimation; not a covariance backend."

import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from .glm import weight_view as nvfp4_view


def weight_view(w, s, gs):
    """compressed-tensors NVFP4 (packed uint8 + group scale + global scale) or FP8 per-channel."""
    if w.dtype == torch.uint8:
        return nvfp4_view(w, s, gs)
    assert w.dtype == torch.float8_e4m3fn and gs is None, w.dtype
    return (w.float() * s.float()).bfloat16()


class LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, s, gs):
        ctx.save_for_backward(w, s, gs)
        return F.linear(x, weight_view(w, s, gs))

    @staticmethod
    def backward(ctx, grad):
        w, s, gs = ctx.saved_tensors
        return grad.bfloat16() @ weight_view(w, s, gs), None, None, None


class PackedLinear(nn.Module):
    def __init__(self, w, s, gs=None):
        super().__init__()
        for name, t in [("packed_weight", w), ("scale", s), ("global_scale", gs)]:
            self.register_buffer(name, t)
        self.in_features = w.shape[-1] * (2 if w.dtype == torch.uint8 else 1)
        self.out_features = w.shape[0]

    def forward(self, x):
        return LinearFn.apply(x, self.packed_weight, self.scale, self.global_scale)


class Checkpoint:
    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads(
            (self.root / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.handles = {}
        self.used = set()

    def get(self, key, device):
        shard = self.index[key]
        if shard not in self.handles:
            self.handles[shard] = safe_open(
                self.root / shard, framework="pt", device="cpu"
            )
        self.used.add(key)
        return self.handles[shard].get_tensor(key).to(device)


def source_name(name):
    return name if name.startswith("lm_head") else "model.language_model." + name[6:]


@torch.no_grad()
def load(root, device="cuda", *, purpose=None):
    """Dense Qwen3.5 compressed-tensors checkpoint: NVFP4 and FP8-channel linears stay packed."""
    if purpose != "paired_gradient":
        raise RuntimeError(
            "Only paired_gradient purpose is permitted; generation/covariance require native vLLM"
        )
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    cfg = AutoConfig.from_pretrained(root, local_files_only=True).text_config
    cfg._attn_implementation = "eager"
    cfg.tie_word_embeddings = False
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            model = hf.Qwen3_5ForCausalLM(cfg)
    finally:
        torch.set_default_dtype(original_dtype)
    cp = Checkpoint(root)
    began = time.time()
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        source = source_name(name)
        if source + ".weight_packed" in cp.index:
            new = PackedLinear(
                cp.get(source + ".weight_packed", device),
                cp.get(source + ".weight_scale", device),
                cp.get(source + ".weight_global_scale", device),
            )
        elif source + ".weight_scale" in cp.index:
            new = PackedLinear(
                cp.get(source + ".weight", device),
                cp.get(source + ".weight_scale", device),
            )
        else:
            continue
        assert mod.bias is None, (source, "bias unsupported")
        assert (new.out_features, new.in_features) == tuple(mod.weight.shape), source
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, new)
    state = {}
    for name, placeholder in model.state_dict().items():
        module_path = name.rpartition(".")[0]
        if isinstance(model.get_submodule(module_path), PackedLinear):
            continue
        value = cp.get(source_name(name), device)
        if value.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(("Unexpected unquantized weight", name, value.dtype))
        assert value.shape == placeholder.shape, (name, value.shape, placeholder.shape)
        state[name] = value.to(placeholder.dtype)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    assert not unexpected, unexpected
    assert all(
        isinstance(model.get_submodule(n.rpartition(".")[0]), PackedLinear)
        for n in missing
    ), [n for n in missing][:10]
    model.model.rotary_emb = hf.Qwen3_5TextRotaryEmbedding(config=cfg).to(device)
    assert not any(t.is_meta for t in [*model.parameters(), *model.buffers()])
    active = {
        key
        for key in cp.index
        if (key.startswith("lm_head.") or key.startswith("model.language_model."))
        and not key.endswith((".input_global_scale", ".k_scale", ".v_scale"))
    }
    assert not active - cp.used, sorted(active - cp.used)[:30]
    model.eval().requires_grad_(False)
    packed = sum(
        x.numel() * x.element_size()
        for m in model.modules()
        if isinstance(m, PackedLinear)
        for x in (m.packed_weight,)
    )
    print(
        f"PACKED GRADIENT MODEL {len(model.model.layers)} layers, packed {packed / 2**30:.2f} GiB, "
        f"device {torch.cuda.memory_allocated() / 2**30:.2f} GiB, {time.time() - began:.0f}s",
        flush=True,
    )
    assert packed > 0
    return model
