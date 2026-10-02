# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"GDN window-boundary covariance and effective-query observer."

import torch
import torch.nn.functional as F


def factors(q, k, a, b, A_log, dt_bias, scale, rounded_beta):
    q = q.float()
    k = k.float()
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) * scale
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    rep = a.shape[-1] // k.shape[-2]
    q = q.repeat_interleave(rep, dim=-2)
    k = k.repeat_interleave(rep, dim=-2)
    alpha = torch.exp(-A_log.float().exp() * F.softplus(a.float() + dt_bias.float()))
    beta = b.float().sigmoid()
    if rounded_beta:
        beta = beta.to(b.dtype).float()
    return q, k, alpha, beta


class Accumulator:
    def __init__(self, slots, heads, K, device, window=16, tokens=256):
        self.window = window
        self.tokens = tokens
        self.K = K
        self.pos = torch.zeros(slots, device=device, dtype=torch.long)

        self.T = torch.zeros(slots, heads, K, K, device=device)
        self.E = torch.zeros(heads, K, K, device=device, dtype=torch.float64)
        self.C = torch.zeros_like(self.E)
        self.windows = 0
        self.queries = 0

    def reset_slots(self):
        self.pos.zero_()
        self.T.zero_()

    def observe(self, state, q, k, alpha, beta, physical, logical, positions):
        take = positions >= 0
        physical = physical[take]
        logical = logical[take]
        positions = positions[take]
        q = q[take]
        k = k[take]
        alpha = alpha[take]
        beta = beta[take]
        if not physical.numel():
            return
        assert physical.unique().numel() == physical.numel()
        assert bool((physical > 0).all()) and bool((logical < self.pos.numel()).all())
        pos = self.pos[logical]
        assert torch.equal(pos, positions), (pos.tolist(), positions.tolist())
        assert bool((pos < self.tokens).all()), pos.tolist()
        start = pos.remainder(self.window) == 0
        measured = pos >= self.window
        boundary = start & measured
        if bool(boundary.any()):
            s = state[physical[boundary]].float()
            self.E.add_(torch.einsum("bhvk,bhvn->hkn", s, s).double())
            self.windows += int(boundary.sum())
        t = self.T[logical]
        t = torch.where(
            start[:, None, None, None], torch.eye(self.K, device=t.device), t
        )
        tk = torch.matmul(t, k[..., None]).squeeze(-1)
        t = alpha[..., None, None] * (
            t - beta[..., None, None] * tk[..., None] * k[..., None, :]
        )
        self.T[logical] = t
        if bool(measured.any()):
            effective = torch.matmul(t[measured], q[measured, ..., None]).squeeze(-1)
            self.C.add_(torch.einsum("bhk,bhn->hkn", effective, effective).double())
            self.queries += int(measured.sum())
        self.pos[logical] += 1
