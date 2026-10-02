# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"MAMBA2 window-boundary covariance and effective-query observer."

import torch
import torch.nn.functional as F


class Accumulator:
    def __init__(self, slots, heads, k, device, window=16, tokens=256):
        self.window, self.tokens = window, tokens
        self.pos = torch.zeros(slots, device=device, dtype=torch.long)
        self.alpha = torch.ones(slots, heads, device=device, dtype=torch.float32)
        self.E = torch.zeros(heads, k, k, device=device, dtype=torch.float64)
        self.C = torch.zeros_like(self.E)
        self.windows = self.queries = 0

    def reset_slots(self):
        self.pos.zero_()
        self.alpha.fill_(1)

    def observe(self, state, dt, A, C, bias, slots, logical=None, positions=None):
        logical = slots if logical is None else logical
        if positions is not None:
            generated = positions >= 0
            slots = slots[generated]
            logical = logical[generated]
            dt = dt[generated]
            C = C[generated]
            positions = positions[generated]
            if not slots.numel():
                return
        pos = self.pos[logical]
        if positions is not None and not torch.equal(pos, positions):
            raise RuntimeError(
                f"Decode position mismatch: counted={pos.tolist()}, actual={positions.tolist()}, physical={slots.tolist()}"
            )
        if bool((pos >= self.tokens).any()):
            raise RuntimeError(f"Unexpected extra decode token: {pos.tolist()}")
        start = pos.remainder(self.window) == 0
        take = pos >= self.window
        boundary = start & take
        if bool(boundary.any()):
            S = state[slots[boundary]].float()

            self.E.add_(torch.einsum("bhvk,bhvn->hkn", S, S).double())
            self.windows += int(boundary.sum())
        alpha = self.alpha[logical]
        alpha = torch.where(start[:, None], 1.0, alpha)
        decay = torch.exp(F.softplus(dt.float() + bias.float()) * A.float())
        alpha = alpha * decay
        self.alpha[logical] = alpha
        if bool(take.any()):
            heads = state.shape[1]
            q = (
                C[take].float().repeat_interleave(heads // C.shape[1], dim=1)
                * alpha[take, :, None]
            )
            self.C.add_(torch.einsum("bhk,bhn->hkn", q, q).double())
            self.queries += int(take.sum())
        self.pos[logical] += 1
