# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Shared NLL/backward loop and recurrence-specific raw-readout hooks."""

import importlib

import torch
import torch.nn.functional as F

from ..core.basis import fingerprint
from .models import load_model
from .storage import atomic_json, atomic_torch, load


class KDABinding:
    def __init__(self, model, basis, config):
        from fla.ops.kda import chunk_kda

        from .hooks.kda import Collector

        layer_ids = basis.get("layer_ids")
        if layer_ids is None:
            raise ValueError("KDA basis requires original layer_ids")
        self.collector = Collector(
            dict(zip(layer_ids, basis["omega"])),
            config["geometry"]["window"],
            config["paired"]["warmup_tokens"],
        )
        layers = getattr(model, "layers", None)
        if layers is None:
            raise ValueError("KDA gradient binding expects GLM text model.layers")
        self.module = importlib.import_module(
            type(layers[layer_ids[0]].self_attn).__module__
        )
        self.original = self.module.chunk_kimi_delta_attention

        def attention(q, k, v, g, beta, **kwargs):
            out, state = chunk_kda(
                q, k, v, g=g, beta=beta.float(), safe_gate=True, **kwargs
            )
            self.collector.observe(q, k, v, g, beta, out)
            return out, state

        self.module.chunk_kimi_delta_attention = attention
        self.handles = [
            layers[li].self_attn.register_forward_pre_hook(
                lambda mod, args, li=li: setattr(self.collector, "current", li)
            )
            for li in layer_ids
        ]
        self.layer_ids = layer_ids

    def start(self):
        self.collector.curves = {}

    def result(self, nsteps):
        if sorted(self.collector.curves) != sorted(self.layer_ids):
            raise RuntimeError("Not all KDA layers supplied a raw-readout gradient")
        keys = (
            "output_error_sum",
            "joint_dot_sq_sum",
            "scalar_grad_output_error_sum",
            "joint_dot_sum",
            "grad_sq_sum",
        )
        return {
            k: torch.stack([self.collector.curves[i][k] for i in self.layer_ids])
            for k in keys
        }

    def close(self):
        self.module.chunk_kimi_delta_attention = self.original
        for handle in self.handles:
            handle.remove()


class ScanBinding:
    def __init__(self, model, basis, config):
        family = config["model"]["family"]
        if family == "mamba2":
            from .hooks.mamba2 import JointCollector as Collector

            attrs = (
                "ssm_state_size",
                "n_groups",
                "num_heads",
                "head_dim",
                "in_proj",
                "norm",
                "A_log",
            )
        else:
            from .hooks.gdn import GDNJointCollector as Collector

            attrs = (
                "num_v_heads",
                "num_k_heads",
                "head_k_dim",
                "head_v_dim",
                "in_proj_qkv",
                "norm",
            )
        mixers = [m for m in model.modules() if all(hasattr(m, key) for key in attrs)]
        if "layer_ids" in basis:
            actual = [getattr(m, "layer_idx", None) for m in mixers]
            if actual != basis["layer_ids"]:
                raise ValueError(
                    f"Native/HF recurrent layer order differs: {actual} vs {basis['layer_ids']}"
                )
        omega = basis["omega"]
        L, groups, rank, K = omega.shape
        if len(mixers) != L:
            raise ValueError(f"Found {len(mixers)} mixers but basis has {L} layers")
        opts = config.get("gradient", {})
        self.collector = Collector(
            mixers,
            omega,
            window=config["geometry"]["window"],
            warmup=config["paired"]["warmup_tokens"],
            mmax=rank,
            rank_tol=1e-5,
            recurrence_atol=opts.get(
                "recurrence_atol", 1.0 if family == "mamba2" else 2.0
            ),
            recurrence_rtol=opts.get(
                "recurrence_rtol", 0.5 if family == "mamba2" else 0.05
            ),
        )

    def start(self):
        self.before = {k: v.clone() for k, v in self.collector.sums.items()}

    def result(self, nsteps):
        self.collector.finish_sequence(nsteps)
        return {k: v - self.before[k] for k, v in self.collector.sums.items()}

    def close(self):
        self.collector.close()


def collect(config, root, identity):
    basis = load(root / "basis/omega.pt")
    tokens = load(root / "data/allocation_tokens.pt")["paired_validation_token_ids"]
    paired = config["paired"]
    if tokens.shape != (paired["sequences"], paired["sequence_length"] + 1):
        raise ValueError("Paired token shape differs from configuration")
    holder, binding = None, None
    all_stats = []
    nsteps = paired["sequence_length"] - paired["warmup_tokens"]
    try:
        for i, row in enumerate(tokens):
            path = root / "progress/paired" / f"{i:06d}.pt"
            if path.exists():
                saved = load(path)
                if saved["identity"] != identity:
                    raise ValueError("Paired checkpoint identity changed")
                stats = saved["stats"]
            else:
                if holder is None:
                    holder = load_model(config)
                    atomic_json(holder.audit, root / "progress/gradient_audit.json")
                    custom = config["gradient"].get("binding")
                    if custom:
                        module, name = custom.split(":", 1)
                        cls = getattr(importlib.import_module(module), name)
                    else:
                        cls = (
                            KDABinding
                            if config["model"]["family"] == "kda"
                            else ScanBinding
                        )
                    binding = cls(holder.model, basis, config)
                binding.start()
                logits = holder.logits(row[None, :-1])
                target = row[None, 1:].to(logits.device)
                loss = F.cross_entropy(
                    logits.flatten(0, 1).float(), target.flatten(), reduction="sum"
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite calibration NLL")
                loss.backward()
                stats = binding.result(nsteps)
                if not all(torch.isfinite(v).all() for v in stats.values()):
                    raise RuntimeError("Nonfinite paired statistics")
                if not bool((stats["grad_sq_sum"].sum(-1) > 0).all()):
                    raise RuntimeError("Missing/zero raw-readout gradient in a layer")
                atomic_torch(
                    dict(
                        identity=identity,
                        stats=stats,
                        nll=float(loss) / paired["sequence_length"],
                    ),
                    path,
                )
                del logits, loss
            all_stats.append(stats)
            print(f"paired: {i + 1}/{len(tokens)}", flush=True)
    finally:
        if binding is not None:
            binding.close()
    keys = set(all_stats[0])
    if any(set(s) != keys for s in all_stats):
        raise RuntimeError("Paired checkpoint schema changed")
    out = {k: torch.stack([s[k] for s in all_stats]).sum(0) for k in sorted(keys)}
    for key in ("output_error_sum", "joint_dot_sq_sum"):
        out[f"per_sequence_{key}"] = torch.stack([s[key].float() for s in all_stats])
    out.update(
        joint_nseq=len(tokens),
        joint_nstep=len(tokens) * nsteps,
        meta=dict(
            schema_version=1,
            coefficient_model="full-gram",
            omega_granularity="group",
            basis_fingerprint=fingerprint(basis["omega"]),
            identity=identity,
            storage_rounding_in_score=False,
            exclude_flush=False,
        ),
    )
    if "layer_ids" in basis:
        out["layer_ids"] = basis["layer_ids"]
    atomic_torch(out, root / "statistics/paired_scores.pt")
