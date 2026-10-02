# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"Packed GLM loader restricted to paired gradient estimation; not a covariance backend."

import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F


def weight_view(w, s, gs):
    from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
    from compressed_tensors.quantization.lifecycle.forward import dequantize

    x = unpack_fp4_from_uint8(w, w.shape[0], w.shape[1] * 2)
    return dequantize(x_q=x, scale=s.to(x.dtype), global_scale=gs, dtype=torch.bfloat16)


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
    def __init__(self, w, s, gs):
        super().__init__()
        for name, t in [("packed_weight", w), ("scale", s), ("global_scale", gs)]:
            self.register_buffer(name, t)
        self.in_features = w.shape[-1] * 2
        self.out_features = w.shape[0]

    def forward(self, x):
        return LinearFn.apply(x, self.packed_weight, self.scale, self.global_scale)


class PackedExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.swiglu_limit = config.swiglu_limit

    def load_projection(self, proj, get, prefix, device):
        for suffix, attr in [
            ("weight_packed", "w"),
            ("weight_scale", "s"),
            ("weight_global_scale", "gs"),
        ]:
            first = get(f"{prefix}.0.{proj}.{suffix}", "cpu")
            dest = torch.empty(
                (self.num_experts, *first.shape), device=device, dtype=first.dtype
            )
            for e in range(self.num_experts):
                dest[e].copy_(get(f"{prefix}.{e}.{proj}.{suffix}", "cpu"))
            self.register_buffer(f"{proj}_{attr}", dest, persistent=False)

    def linear(self, x, proj, e):
        return LinearFn.apply(
            x,
            getattr(self, proj + "_w")[e],
            getattr(self, proj + "_s")[e],
            getattr(self, proj + "_gs")[e],
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = (mask.sum((-1, -2)) > 0).nonzero().flatten().tolist()
        for e in hit:
            pos, tok = torch.where(mask[e])
            x = hidden_states[tok]
            gate = self.linear(x, "gate_proj", e).clamp(max=self.swiglu_limit)
            up = self.linear(x, "up_proj", e).clamp(
                min=-self.swiglu_limit, max=self.swiglu_limit
            )
            current = (
                self.linear(F.silu(gate) * up, "down_proj", e)
                * top_k_weights[tok, pos, None]
            )
            final.index_add_(0, tok, current.to(final.dtype))
        return final


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

    def unquantized(self, key, device):
        value = self.get(key, device)
        if value.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(("Unexpected unquantized weight", key, value.dtype))
        return value


def source_name(name):
    for target, source in [("attn_hc", "hc_attn"), ("ffn_hc", "hc_ffn")]:
        for suffix in ["fn", "base", "scale"]:
            name = name.replace(f"{target}.{suffix}", f"{source}_{suffix}")
    return name.replace("self_attn.forget_gate.", "self_attn.")


@torch.no_grad()
def load(root, device="cuda", *, purpose=None):
    if purpose != "paired_gradient":
        raise RuntimeError(
            "Only paired_gradient purpose is permitted; generation/covariance require native vLLM NVFP4"
        )
    from transformers import AutoConfig
    from transformers.models.glm5_next import modeling_glm5_next as hf

    cfg = AutoConfig.from_pretrained(root, local_files_only=True).text_config
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model = hf.Glm5NextTextModel(cfg)
    cp = Checkpoint(root)
    began = time.time()
    for i, layer in enumerate(model.layers):
        prefix = f"model.language_model.layers.{i}."
        if hasattr(layer.mlp, "experts"):
            layer.mlp.experts = PackedExperts(cfg)
        for name, mod in list(layer.named_modules()):
            source = prefix + source_name(name)
            if (
                not isinstance(mod, nn.Linear)
                or source + ".weight_packed" not in cp.index
            ):
                continue
            assert mod.bias is None, (source, "bias unsupported")
            new = PackedLinear(
                cp.get(source + ".weight_packed", device),
                cp.get(source + ".weight_scale", device),
                cp.get(source + ".weight_global_scale", device),
            )
            parent, _, leaf = name.rpartition(".")
            setattr(layer.get_submodule(parent) if parent else layer, leaf, new)
        state = {}
        for name, placeholder in layer.state_dict().items():
            module_path, _, attr = name.rpartition(".")
            owner = layer.get_submodule(module_path) if module_path else layer
            if isinstance(owner, PackedLinear):
                continue
            if name == "self_attn.conv1d.weight":
                value = torch.cat(
                    [
                        cp.unquantized(prefix + f"self_attn.{q}_conv1d.weight", device)
                        for q in ["q", "k", "v"]
                    ]
                )
            else:
                value = cp.get(prefix + source_name(name), device)
            assert value.shape == placeholder.shape, (
                i,
                name,
                value.shape,
                placeholder.shape,
            )
            state[name] = value
        missing, unexpected = layer.load_state_dict(state, strict=False, assign=True)
        assert not unexpected and all(
            isinstance(layer.get_submodule(n.rpartition(".")[0]), PackedLinear)
            for n in missing
        ), (missing, unexpected)
        if isinstance(
            layer.mlp.experts if hasattr(layer.mlp, "experts") else None, PackedExperts
        ):
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                layer.mlp.experts.load_projection(
                    proj, cp.get, prefix + "mlp.experts", device
                )
        print(
            f"PACKED GRADIENT MODEL layer {i + 1}/{len(model.layers)}, {torch.cuda.memory_allocated() / 2**30:.2f} GiB, {time.time() - began:.0f}s",
            flush=True,
        )
    model.embed_tokens.load_state_dict(
        {"weight": cp.unquantized("model.language_model.embed_tokens.weight", device)},
        assign=True,
    )
    model.norm.load_state_dict(
        {"weight": cp.unquantized("model.language_model.norm.weight", device)},
        assign=True,
    )
    head = cp.unquantized("lm_head.weight", device)
    active = {
        key
        for key in cp.index
        if (key == "lm_head.weight" or key.startswith("model.language_model."))
        and not key.startswith(f"model.language_model.layers.{len(model.layers)}.")
        and not key.endswith(".input_global_scale")
    }
    assert not active - cp.used, sorted(active - cp.used)[:30]
    assert not any(p.is_meta for p in model.parameters())
    model.eval().requires_grad_(False)
    packed = sum(
        x.numel() * x.element_size() for x in model.buffers() if x.dtype == torch.uint8
    )
    assert packed > 0, packed
    return (
        model,
        head,
        dict(
            purpose=purpose,
            packed_uint8_bytes=packed,
            used_tensors=len(cp.used),
            unconsumed=0,
            device_bytes=torch.cuda.memory_allocated(),
            native_backward=False,
        ),
    )


def forward(model, head, ids, grad=True):
    device = head.device
    hidden = model.embed_tokens(ids.to(device)).detach().requires_grad_(grad)
    hidden = hidden.unsqueeze(2).expand(-1, -1, model.config.hc_mult, -1).contiguous()
    previous = None
    for layer in model.layers:
        hidden, previous = layer(
            hidden,
            attention_mask=torch.ones(ids.shape, device=device, dtype=torch.bool),
            position_ids=torch.arange(ids.shape[1], device=device)[None],
            prev_topk_indices=previous,
            use_cache=False,
        )
    return F.linear(model.norm(model.hc_head(hidden)), head)
