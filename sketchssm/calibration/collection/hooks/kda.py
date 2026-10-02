# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""KDA paired residual and output-gradient collector."""

import torch

from .kda_statistics import features, paired_curves, prefix_qr


class Collector:
    def __init__(self, basis, window=16, warmup=128):
        self.window = window
        self.warmup = warmup
        if warmup % window:
            raise ValueError("warmup must align to windows")
        self.basis = basis
        self.current = None
        self.curves = {}
        self.enabled = False
        self.max_transition_error = 0.0
        self.max_fla_error = 0.0

    def observe(self, q, k, v, g, beta, output):
        layer = self.current
        assert layer in self.basis
        with torch.no_grad():
            starts, x, boundary, dense, error = features(
                q, k, v, g, beta, window=self.window
            )
            self.max_transition_error = max(self.max_transition_error, error)
            rel = float(
                (dense - output[0].float()).norm() / dense.norm().clamp_min(1e-30)
            )
            self.max_fla_error = max(self.max_fla_error, rel)
            assert torch.isfinite(torch.tensor(rel)) and rel < 0.05, (layer, rel)
            starts, boundary = (
                starts[self.warmup // self.window :],
                boundary[self.warmup // self.window :],
            )
            omega = self.basis[layer].to(q.device)
            basis_q, rejected = prefix_qr(
                starts.double() @ omega.double().transpose(-1, -2)[None], tol=1e-05
            )
            boundary = boundary.double()

        def backward(grad):
            assert layer not in self.curves, ("duplicate gradient", layer)
            grad = grad[0, self.warmup :].reshape(boundary.shape).double()
            stats = paired_curves(grad, boundary, basis_q)
            stats["grad_sq_sum"] = grad.square().sum((0, 1, 3)).cpu()
            self.curves[layer] = stats

        output.register_hook(backward)
