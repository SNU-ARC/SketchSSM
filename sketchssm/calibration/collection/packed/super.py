# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"Packed Super loader for BF16-activation input-gradient estimation; not native NVFP4 backward."

import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F


def weight_view(w, s, s2):
    if w.dtype == torch.uint8:
        levels = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device=w.device,
        )
        val = torch.stack(
            (levels[(w & 15).long()], levels[(w >> 4).long()]), -1
        ).flatten(-2)
        return (val * s.float().repeat_interleave(16, -1) * s2.float()).bfloat16()
    assert w.dtype == torch.float8_e4m3fn, w.dtype
    return (w.float() * s.float()).bfloat16()


class PackedLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, s, s2):
        ctx.save_for_backward(w, s, s2)
        return F.linear(x, weight_view(w, s, s2))

    @staticmethod
    def backward(ctx, g):
        w, s, s2 = ctx.saved_tensors
        return g.to(torch.bfloat16) @ weight_view(w, s, s2), None, None, None


class PackedLinear(nn.Module):
    def __init__(self, w, s, s2, bias=None):
        super().__init__()
        for name, t in [("packed_weight", w), ("scale", s), ("scale2", s2)]:
            self.register_buffer(name, t)
        self.register_buffer("bias", bias)
        self.in_features = w.shape[-1] * (2 if w.dtype == torch.uint8 else 1)
        self.out_features = w.shape[0]

    def forward(self, x):
        y = PackedLinearFn.apply(x, self.packed_weight, self.scale, self.scale2)
        return y if self.bias is None else y + self.bias


class PackedExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.activations import ACT2FN

        self.num_experts = config.n_routed_experts
        self.act_fn = ACT2FN[config.mlp_hidden_act]

    def load_projection(self, proj, get, source, device):
        for suffix, attr in [
            ("weight", "w"),
            ("weight_scale", "s"),
            ("weight_scale_2", "s2"),
        ]:
            first = get(f"{source}.0.{proj}.{suffix}")
            dest = torch.empty(
                (self.num_experts, *first.shape), dtype=first.dtype, device=device
            )
            for e in range(self.num_experts):
                dest[e].copy_(get(f"{source}.{e}.{proj}.{suffix}"))
            self.register_buffer(f"{proj}_{attr}", dest)

    def linear(self, x, proj, e):
        return PackedLinearFn.apply(
            x,
            getattr(self, proj + "_w")[e],
            getattr(self, proj + "_s")[e],
            getattr(self, proj + "_s2")[e],
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states, dtype=top_k_weights.dtype)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = (mask.sum((-1, -2)) > 0).nonzero().flatten().tolist()
        for e in hit:
            pos, tok = torch.where(mask[e])
            h = self.act_fn(self.linear(hidden_states[tok], "up_proj", e))
            y = self.linear(h, "down_proj", e) * top_k_weights[tok, pos, None]
            out.index_add_(0, tok, y.to(out.dtype))
        return out.to(hidden_states.dtype)


def load(model_dir, device="cuda", verbose=True, *, purpose=None):
    if purpose != "paired_gradient":
        raise RuntimeError("Packed reconstruction is restricted to paired_gradient")
    from transformers import AutoConfig
    from transformers.models.nemotron_h import modeling_nemotron_h as M

    started = time.time()
    root = Path(model_dir)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    handles = {}
    used = set()

    def get(k):
        used.add(k)
        shard = index[k]
        if shard not in handles:
            handles[shard] = safe_open(root / shard, framework="pt", device="cpu")
        return handles[shard].get_tensor(k)

    def hf(k):
        return "model." + k[len("backbone.") :] if k.startswith("backbone.") else k

    def source(k):
        return "backbone." + k[len("model.") :] if k.startswith("model.") else k

    cfg = AutoConfig.from_pretrained(model_dir)
    cfg.quantization_config = None
    old = M.NemotronHExperts
    old_dtype = torch.get_default_dtype()
    try:
        M.NemotronHExperts = PackedExperts
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            model = M.NemotronHForCausalLM(cfg)
    finally:
        M.NemotronHExperts = old
        torch.set_default_dtype(old_dtype)

    quantized = 0
    for name, m in list(model.named_modules()):
        if not isinstance(m, nn.Linear):
            continue
        pre = source(name)
        if pre + ".weight_scale" not in index:
            continue
        w = get(pre + ".weight").to(device)
        assert w.dtype in (torch.uint8, torch.float8_e4m3fn), (name, w.dtype)
        s = get(pre + ".weight_scale").to(device)
        s2 = (
            get(pre + ".weight_scale_2").to(device)
            if pre + ".weight_scale_2" in index
            else torch.ones((), device=device)
        )
        bias = get(pre + ".bias").to(device) if pre + ".bias" in index else None
        parent, leaf = name.rsplit(".", 1) if "." in name else ("", name)
        setattr(model.get_submodule(parent), leaf, PackedLinear(w, s, s2, bias))
        quantized += 1
    expert_count = 0
    for name, m in model.named_modules():
        if not isinstance(m, PackedExperts):
            continue
        for proj in ("up_proj", "down_proj"):
            m.load_projection(proj, get, source(name), device)
        expert_count += 1
        if verbose:
            print(
                f"[packed-super] experts {expert_count} packed; {time.time() - started:.0f}s",
                flush=True,
            )
    assert quantized or expert_count, "No packed weights loaded"

    for name, p in list(model.named_parameters()):
        if not p.is_meta:
            continue
        key = source(name)
        t = get(key).to(device)
        assert t.dtype not in (torch.uint8, torch.float8_e4m3fn), key
        parent, leaf = name.rsplit(".", 1) if "." in name else ("", name)
        setattr(model.get_submodule(parent), leaf, nn.Parameter(t, requires_grad=False))
    for name, b in list(model.named_buffers()):
        if not b.is_meta:
            continue
        key = source(name)
        if key in index:
            parent, leaf = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), leaf, get(key).to(device))
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = M.NemotronHRotaryEmbedding(config=cfg).to(device)
    remaining = [
        n
        for n, t in list(model.named_parameters()) + list(model.named_buffers())
        if t.is_meta
    ]
    assert not remaining, remaining
    allowed = (".input_scale", ".k_scale", ".v_scale")
    extra = [
        k
        for k in index
        if k not in used and not k.startswith("mtp.") and not k.endswith(allowed)
    ]
    assert not extra, extra[:20]
    for p in model.parameters():
        p.requires_grad_(False)
    packed_bytes = sum(b.numel() for b in model.buffers() if b.dtype == torch.uint8)
    assert packed_bytes > 0
    model.eval()
    model.packed_audit = dict(
        expert_layers=expert_count,
        quantized_linears=quantized,
        packed_uint8_bytes=packed_bytes,
        full_model_bf16_materialized=False,
        gradient_arithmetic="temporary per-linear BF16 weights and BF16 activations; packed autograd tape",
    )
    if verbose:
        print("[packed-super]", model.packed_audit, flush=True)
    return model
