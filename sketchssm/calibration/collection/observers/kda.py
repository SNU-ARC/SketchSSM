# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"KDA window-boundary covariance and effective-query observer."

import torch


def factors(q, k, raw_gate, raw_beta, a_log, bias, lower=-5.0):
    q = q.float()
    k = k.float()
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() / q.shape[-1] ** 0.5
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    alpha = torch.exp(
        lower
        * torch.sigmoid(
            a_log.float().reshape(1, -1, 1).exp()
            * (raw_gate.float() + bias.float().reshape(1, -1, q.shape[-1]))
        )
    )
    return q, k, alpha, raw_beta.float().sigmoid()


class Accumulator:
    def __init__(
        self, slots, heads, K, device, window=16, tokens=256, physical_slots=None
    ):
        self.window = window
        self.tokens = tokens
        self.K = K
        self.slot_map = (
            None
            if physical_slots is None
            else torch.full((physical_slots,), -1, device=device, dtype=torch.long)
        )
        self.next_slot = 0
        self.pos = torch.zeros(slots, device=device, dtype=torch.long)
        self.T = torch.zeros(slots, heads, K, K, device=device)
        self.E = torch.zeros(heads, K, K, device=device, dtype=torch.float64)
        self.C = torch.zeros_like(self.E)
        self.windows = 0
        self.queries = 0
        self.min_physical = None

    def reset_slots(self):
        self.pos.zero_()
        self.T.zero_()
        self.next_slot = 0
        if self.slot_map is not None:
            self.slot_map.fill_(-1)

    def observe(self, state, q, k, alpha, beta, physical):
        valid = physical >= 0
        physical = physical[valid]
        q = q[valid]
        k = k[valid]
        alpha = alpha[valid]
        beta = beta[valid]
        if not physical.numel():
            return
        assert physical.unique().numel() == physical.numel()
        logical = physical
        if self.slot_map is not None:
            unseen = self.slot_map[physical] < 0
            n = int(unseen.sum())
            assert self.next_slot + n <= self.pos.numel(), (
                "Too many live calibration slots"
            )
            self.slot_map[physical[unseen]] = torch.arange(
                self.next_slot, self.next_slot + n, device=physical.device
            )
            self.next_slot += n
            logical = self.slot_map[physical]
        pos = self.pos[logical]
        assert bool((pos < self.tokens).all()), pos.tolist()
        self.min_physical = (
            min(int(physical.min()), self.min_physical)
            if self.min_physical is not None
            else int(physical.min())
        )
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

        t = t * alpha[..., None, :]
        tk = torch.matmul(t, k[..., None]).squeeze(-1)
        t = t - beta[..., None, None] * tk[..., None] * k[..., None, :]
        self.T[logical] = t
        if bool(measured.any()):
            effective = torch.matmul(t[measured], q[measured, ..., None]).squeeze(-1)
            self.C.add_(torch.einsum("bhk,bhn->hkn", effective, effective).double())
            self.queries += int(measured.sum())
        self.pos[logical] += 1
