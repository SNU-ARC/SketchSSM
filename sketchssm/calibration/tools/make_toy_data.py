# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Small synthetic tensors for checking the calibration/allocation interface."""
import argparse
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    L, B, N, H, V, K, W = 1, 2, 3, 4, 8, 128, 16
    s = torch.randn(L, B, N, H, V, K, dtype=torch.float64)
    q = torch.randn(L, B, N, H, K, W, dtype=torch.float64)
    g = torch.randn(L, B, N, H, V, W, dtype=torch.float64)
    torch.save(dict(state=s, effective_query=q, gradient=g), a.out / 'trace.pt')
    E = torch.einsum('lbnhvk,lbnhvj->lhkj', s, s)
    C = torch.einsum('lbnhkw,lbnhjw->lhkj', q, q)
    torch.save(dict(head_scov=E, head_qcov=C,
                    head_cov_windows=torch.full((L,), B * N),
                    head_cov_queries=torch.full((L,), B * N * W)), a.out / 'covariance.pt')


if __name__ == '__main__':
    main()
