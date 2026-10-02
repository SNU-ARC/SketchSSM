# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"Packed Qwen loader restricted to paired gradient estimation; not a covariance backend."

import glob
import os
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from .fp4 import unpack_fp4

_EXP = re.compile(
    r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale|weight_scale_2|input_scale)$"
)
_SHARD = re.compile(r"^(.*\.ngram_embedding)\.shard_(\d+)\.weight$")


def dequant(w_u8, s_e4m3, s2, group=16):
    """NVFP4 to BF16: nibble value times block scale times global scale."""
    w = unpack_fp4(w_u8)
    s = s_e4m3.float().repeat_interleave(group, dim=-1)
    return (w * s * s2.float()).to(torch.bfloat16)


class QLinearFn(torch.autograd.Function):
    """Packed frozen-weight linear; backward reconstructs only this weight for input gradients."""

    @staticmethod
    def forward(ctx, x, w_u8, s, s2):
        W = dequant(w_u8, s, s2)
        ctx.save_for_backward(w_u8, s, s2)
        return F.linear(x, W)

    @staticmethod
    def backward(ctx, g):
        w_u8, s, s2 = ctx.saved_tensors
        W = dequant(w_u8, s, s2)
        return g.to(W.dtype) @ W, None, None, None


class NVFP4Experts(nn.Module):
    """Packed experts with the model-compatible routing interface."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = F.silu
        self._loaded = False

    def load_packed(self, packs, device):
        """packs[e] = {gate_proj: (w,s,s2), up_proj: ..., down_proj: ...}"""
        E = self.num_experts
        assert len(packs) == E, (len(packs), E)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            w = torch.stack([packs[e][proj][0] for e in range(E)]).to(device)
            s = torch.stack([packs[e][proj][1] for e in range(E)]).to(device)
            s2 = torch.stack([packs[e][proj][2].reshape(()) for e in range(E)]).to(
                device
            )
            self.register_buffer(f"{proj}_w", w, persistent=False)
            self.register_buffer(f"{proj}_s", s, persistent=False)
            self.register_buffer(f"{proj}_s2", s2, persistent=False)
        self._loaded = True

    def _lin(self, x, proj, e):
        return QLinearFn.apply(
            x,
            getattr(self, f"{proj}_w")[e],
            getattr(self, f"{proj}_s")[e],
            getattr(self, f"{proj}_s2")[e],
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        assert self._loaded, "Packed weights must be loaded before forward"
        out = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
        for e in hit:
            pos, tok = torch.where(mask[e])
            x = hidden_states[tok]
            h = self.act_fn(self._lin(x, "gate_proj", e)) * self._lin(x, "up_proj", e)
            y = self._lin(h, "down_proj", e) * top_k_weights[tok, pos, None]
            out.index_add_(0, tok, y.to(out.dtype))
        return out


class FP8Embedding(nn.Module):
    """FP8 embedding lookup with BF16 outputs; keep the table in FP8 storage."""

    def __init__(self):
        super().__init__()
        self.weight = None
        self.scale = 1.0

    def set(self, w, scale):
        self.weight = w
        self.scale = float(scale)

    def forward(self, ids):
        e = (
            F.embedding(ids, self.weight.view(torch.uint8))
            .view(torch.float8_e4m3fn)
            .to(torch.bfloat16)
        )
        return e * self.scale


def _files(model_dir, pat):
    return sorted(glob.glob(os.path.join(model_dir, pat)))


def load(model_dir, device="cuda", verbose=True, *, purpose=None):
    if purpose != "paired_gradient":
        raise RuntimeError(
            "Packed HF loader is restricted to paired-gradient estimation. "
            "Native NVFP4 generation/covariance must use vLLM; HF forward "
            "covariance and full-model BF16 expansion are retired."
        )
    from transformers import AutoConfig
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M

    t0 = time.time()
    cfg = AutoConfig.from_pretrained(model_dir)
    tc = cfg.text_config
    tc.quantization_config = None

    original_experts = M.Qwen4ExpTextExperts
    original_dtype = torch.get_default_dtype()
    try:
        M.Qwen4ExpTextExperts = NVFP4Experts
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            model = M.Qwen4ExpForCausalLM(tc)
    finally:
        M.Qwen4ExpTextExperts = original_experts
        torch.set_default_dtype(original_dtype)

    for mod in model.modules():
        if isinstance(mod, M.Qwen4ExpTextNGramEmbedding):
            mod.ngram_embedding = FP8Embedding()
    model.to_empty(device=device)

    model.model.rotary_emb = M.Qwen4ExpTextRotaryEmbedding(config=tc).to(device)
    for mod in model.modules():
        if isinstance(mod, M.Qwen4ExpTextNGramEmbedding):
            mod.ngram_heads_vocab_sizes = torch.tensor(
                mod.head_vocab_sizes, dtype=torch.long, device=device
            )
            mod.ngram_heads_offsets = torch.tensor(
                mod.head_offsets, dtype=torch.long, device=device
            )

    sd = {}
    for f in _files(model_dir, "model-bf16-*.safetensors"):
        with safe_open(f, "pt", device="cpu") as h:
            for k in h.keys():
                if k.startswith("mtp.") or k.startswith("model.visual."):
                    continue
                kk = k.replace("model.language_model.", "model.", 1)
                sd[kk] = h.get_tensor(k)
    miss, unexp = model.load_state_dict(sd, strict=False)
    miss = [
        k
        for k in miss
        if ".mlp.experts." not in k and "ngram_embedding.weight" not in k
    ]
    assert not unexp, f"Unexpected checkpoint keys {len(unexp)}: {unexp[:5]}"
    assert not miss, f"Missing model keys {len(miss)}: {miss[:8]}"
    if verbose:
        print(f"[q38fn-hf] bf16 {len(sd)} tensors  {time.time() - t0:.0f}s", flush=True)
    del sd

    shards, scales = {}, {}
    for f in _files(model_dir, "model-plefp8-*.safetensors"):
        with safe_open(f, "pt", device="cpu") as h:
            for k in h.keys():
                if k.endswith(".ngram_embedding.weight_scale"):
                    scales[k[: -len(".weight_scale")]] = (
                        h.get_tensor(k).float().reshape(()).item()
                    )
                    continue
                m = _SHARD.match(k)
                assert m, k
                shards.setdefault(m.group(1), {})[int(m.group(2))] = h.get_tensor(k)
    for pre, parts in shards.items():
        w = torch.cat([parts[i] for i in range(len(parts))], 0).to(device)
        mod = model.get_submodule(pre.replace("model.language_model.", "model.", 1))
        assert isinstance(mod, FP8Embedding)
        mod.set(w, scales[pre])
        if verbose:
            print(
                f"[q38fn-hf] PLE {pre.split('.layers.')[1].split('.')[0]}layers  {tuple(w.shape)} fp8  "
                f"{w.numel() / 2**30:.1f} GiB  scale {scales[pre]:.3g}  {time.time() - t0:.0f}s",
                flush=True,
            )
    del shards

    packs = {}
    for f in _files(model_dir, "layer-*-experts-*.safetensors"):
        with safe_open(f, "pt", device="cpu") as h:
            for k in h.keys():
                m = _EXP.match(k)
                assert m, k
                if m.group(4) == "input_scale":
                    continue
                packs.setdefault(m.group(1), {}).setdefault(
                    int(m.group(2)), {}
                ).setdefault(m.group(3), {})[m.group(4)] = h.get_tensor(k)
    for pre, ex in packs.items():
        mod = model.get_submodule(pre.replace("model.language_model.", "model.", 1))
        assert isinstance(mod, NVFP4Experts), pre
        mod.load_packed(
            {
                e: {
                    p: (d["weight"], d["weight_scale"], d["weight_scale_2"])
                    for p, d in ex[e].items()
                }
                for e in ex
            },
            device,
        )
    if verbose:
        print(
            f"[q38fn-hf] experts {len(packs)} layers  {time.time() - t0:.0f}s  "
            f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB",
            flush=True,
        )
    del packs
    model.eval().requires_grad_(False)
    assert all(m._loaded for m in model.modules() if isinstance(m, NVFP4Experts))
    return model
