# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Mamba scan API bridge; the reference implementation is for CPU module gates."""

import torch
import torch.nn.functional as F


def scan(
    module,
    x,
    dt,
    A,
    B,
    C,
    *,
    chunk_size,
    D,
    dt_bias,
    dt_softplus=True,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
):
    function = getattr(module, "mamba2_chunk_scan", None)
    if function is None:
        function = getattr(module, "mamba_chunk_scan_combined", None)
    if function is not None:
        return function(
            x,
            dt,
            A,
            B,
            C,
            chunk_size=chunk_size,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            return_final_states=return_final_states,
        )
    if x.is_cuda:
        raise RuntimeError(
            "Install the model-compatible Mamba scan kernel for gradient collection"
        )
    batch, tokens, heads, value = x.shape
    groups, key = B.shape[-2:]
    b = B.float().repeat_interleave(heads // groups, dim=2)
    c = C.float().repeat_interleave(heads // groups, dim=2)
    delta = dt.float() + dt_bias.float()
    if dt_softplus:
        delta = F.softplus(delta)
    delta = delta.clamp(*dt_limit)
    state = torch.zeros(batch, heads, value, key, device=x.device, dtype=torch.float32)
    outputs = []
    for t in range(tokens):
        state = state * torch.exp(delta[:, t] * A.float())[:, :, None, None]
        state = (
            state
            + delta[:, t, :, None, None]
            * x[:, t, :, :, None].float()
            * b[:, t, :, None, :]
        )
        outputs.append(
            torch.einsum("bhvk,bhk->bhv", state, c[:, t])
            + D.float()[None, :, None] * x[:, t]
        )
    output = torch.stack(outputs, 1).to(x.dtype)
    return (output, state) if return_final_states else output
