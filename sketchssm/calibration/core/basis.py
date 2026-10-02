# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Fit group-shared sketch bases from normalized per-head covariances."""
import hashlib
import torch


def fingerprint(omega):
    """Content identity independent of checkpoint filenames and storage location."""
    rows = omega.detach().cpu().float().contiguous()
    h = hashlib.sha256(str(tuple(rows.shape)).encode())
    h.update(rows.numpy().tobytes())
    return h.hexdigest()


def solve(state_cov, query_cov, rank, ridge=0.1):
    """Return ordered basis rows, eigenvalues and the regularized state metric."""
    E = state_cov.double()
    C = query_cov.double()
    if E.shape != C.shape or E.ndim < 2 or E.shape[-1] != E.shape[-2]:
        raise ValueError('Covariances must have matching (..., K, K) shapes')
    K = E.shape[-1]
    if not 1 <= rank <= K or not 0 < ridge < float('inf'):
        raise ValueError('Require 1 <= rank <= K and positive finite ridge')
    if not torch.isfinite(E).all() or not torch.isfinite(C).all():
        raise ValueError('Non-finite covariance')
    E = (E + E.transpose(-1, -2)) * 0.5
    C = (C + C.transpose(-1, -2)) * 0.5
    E = E + ridge * E.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] / K * torch.eye(K, dtype=E.dtype, device=E.device)
    ev, U = torch.linalg.eigh(E)
    Eh = (U * ev.clamp_min(0).sqrt().unsqueeze(-2)) @ U.transpose(-1, -2)
    Ei = (U * ev.clamp_min(1e-30).rsqrt().unsqueeze(-2)) @ U.transpose(-1, -2)
    M = Eh @ C @ Eh
    values, vectors = torch.linalg.eigh((M + M.transpose(-1, -2)) * 0.5)
    omega = Ei @ vectors.flip(-1)[..., :rank]
    omega = omega / omega.norm(dim=-2, keepdim=True).clamp_min(1e-12)
    return omega.transpose(-1, -2).float(), values.flip(-1), E


def fit(covariance, groups, rank, ridge=0.1):
    E, C = covariance['head_scov'].double(), covariance['head_qcov'].double()
    if E.shape != C.shape or E.ndim != 4 or E.shape[-1] != E.shape[-2]:
        raise ValueError('Expected matching (layers, heads, K, K) covariances')
    L, H, K, _ = E.shape
    if groups < 1 or H % groups:
        raise ValueError('Native groups must divide the state-head count')
    pooled = []
    for x, key in ((E, 'head_cov_windows'), (C, 'head_cov_queries')):
        count = covariance[key].to(device=x.device, dtype=torch.float64)
        if count.shape != (L,) or not torch.isfinite(count).all() or not (count > 0).all():
            raise ValueError('Each layer must have a positive finite observation count')
        pooled.append((x / count[:, None, None, None]).reshape(L, groups, H // groups, K, K).mean(2))
    omega = solve(*pooled, rank, ridge)[0]
    out = dict(omega=omega, meta=dict(schema_version=1, granularity='group',
        groups=groups, heads_per_group=H // groups, ridge=ridge,
        coordinates='original', gradients_used=False, basis_fingerprint=fingerprint(omega)))
    if 'layer_ids' in covariance:
        out['layer_ids'] = covariance['layer_ids']
    return out
