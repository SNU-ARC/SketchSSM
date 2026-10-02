# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Shared lifecycle and validation for native engine observers."""

import re
from pathlib import Path

import torch

from ..storage import atomic_torch


class CovarianceWorker:
    def configure(self, config):
        self.K = config["geometry"]["key_dim"]
        self.V = config["geometry"]["value_dim"]
        self.W = config["geometry"]["window"]
        self.groups = config["geometry"]["groups"]
        self.tokens = config["generation"]["new_tokens"]
        self.prompt_tokens = config["generation"]["prompt_tokens"]
        self.slots = config["runtime"]["batch_size"]
        self.require_packed = (
            config.get("gradient", {}).get("loader", "ordinary") != "ordinary"
        )
        self._request_indices = {}

    def register_layer(self, name, module):
        layer = getattr(module, "layer_idx", None)
        if layer is None:
            match = re.search(r"(?:layers|blocks)\.(\d+)(?:\.|$)", name)
            if match is None:
                raise ValueError(
                    f"Cannot identify recurrent layer index from {name}; supply a model binding"
                )
            layer = int(match.group(1))
        if int(layer) in self._layer_ids:
            raise ValueError(f"Duplicate recurrent layer index: {layer}")
        self._layer_ids.append(int(layer))

    def check_native_weights(self, packed_bytes, quant_methods):
        if self.require_packed and not packed_bytes:
            raise RuntimeError(
                f"Expected packed native weights, observed {quant_methods}"
            )

    def audit_model(self, model):
        quant = {}
        for m in model.modules():
            method = getattr(m, "quant_method", None)
            if method is not None:
                key = type(method).__name__
                quant[key] = quant.get(key, 0) + 1
        packed = sum(p.numel() for p in model.parameters() if p.dtype == torch.uint8)
        self.check_native_weights(packed, quant)

    def reset_covariance_slots(self):
        self._request_indices.clear()
        for acc in self._cov:
            acc.reset_slots()

    def save_covariance(self, path, completed, batch_size, meta):
        windows = self.tokens // self.W - 1
        queries = self.tokens - self.W
        for acc in self._cov:
            used = acc.pos[acc.pos > 0]
            if used.numel() != batch_size or not bool((used == self.tokens).all()):
                raise RuntimeError(f"Incomplete native trace: {used.tolist()}")
            if acc.windows != completed * windows or acc.queries != completed * queries:
                raise RuntimeError(
                    "Native capture count mismatch; inspect engine decode routing"
                )
        result = dict(
            head_scov=torch.stack([a.E.cpu() for a in self._cov]),
            head_qcov=torch.stack([a.C.cpu() for a in self._cov]),
            head_cov_windows=torch.tensor([a.windows for a in self._cov]),
            head_cov_queries=torch.tensor([a.queries for a in self._cov]),
            processed_sequences=completed,
            meta=meta,
        )
        if hasattr(self, "_layer_ids"):
            result["layer_ids"] = self._layer_ids
        if not all(torch.isfinite(result[k]).all() for k in ("head_scov", "head_qcov")):
            raise RuntimeError("Nonfinite native covariance")
        atomic_torch(result, Path(path))
        return {
            "completed": completed,
            "windows": completed * windows,
            "queries": completed * queries,
        }
