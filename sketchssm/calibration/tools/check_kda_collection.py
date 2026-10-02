# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Small CUDA-only KDA statistics gate; no model weights or vLLM required."""

import json

import torch

from sketchssm.calibration.collection.hooks.kda_statistics import features


def main():
    torch.manual_seed(19)
    H, K, V, T, W = 2, 128, 128, 32, 16
    q, k, v = [
        torch.randn(1, T, H, K, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    g = -torch.rand(1, T, H, K, device="cuda") * 0.1
    beta = torch.rand(1, T, H, device="cuda")
    starts, _, boundary, output, decomposition_error = features(q, k, v, g, beta, W)
    qf = (
        q[0].float()
        / (q[0].float().square().sum(-1, keepdim=True) + 1e-6).sqrt()
        / K**0.5
    )
    kf = k[0].float() / (k[0].float().square().sum(-1, keepdim=True) + 1e-6).sqrt()
    state = torch.zeros(H, V, K, device="cuda")
    states, reads, boundary_reads = [], [], []
    for t in range(T):
        if t % W == 0:
            states.append(state.clone())
            b = state.clone()
        decay = g[0, t].exp()
        state = state * decay[:, None, :]
        erased = torch.einsum("hvk,hk->hv", state, kf[t])
        delta = beta[0, t, :, None] * (v[0, t].float() - erased)
        state = state + delta[:, :, None] * kf[t, :, None, :]
        b = b * decay[:, None, :]
        b = (
            b
            - beta[0, t, :, None, None]
            * torch.einsum("hvk,hk->hv", b, kf[t])[:, :, None]
            * kf[t, :, None, :]
        )
        reads.append(torch.einsum("hvk,hk->hv", state, qf[t]))
        boundary_reads.append(torch.einsum("hvk,hk->hv", b, qf[t]))
    torch.testing.assert_close(starts, torch.stack(states), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(output, torch.stack(reads), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        boundary,
        torch.stack(boundary_reads).reshape(T // W, W, H, V),
        rtol=2e-5,
        atol=2e-6,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "device": torch.cuda.get_device_name(),
                "decomposition_relative_error": decomposition_error,
                "state_max_abs": float((starts - torch.stack(states)).abs().max()),
                "output_max_abs": float((output - torch.stack(reads)).abs().max()),
            }
        )
    )


if __name__ == "__main__":
    main()
