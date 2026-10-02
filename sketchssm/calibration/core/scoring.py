# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Same-token, same-head residual/gradient curves for per-head allocation."""
import torch
from .basis import fingerprint
from ._statistics import _mgs_prefix, _rank_statistics


def score(trace, basis):
    """Full-Gram prefix projection in FP64, independent of inference pivots.

    Uses all supplied calibration queries, matching the Full-Gram collector.
    This is an ideal output-space objective, not a simulation of runtime ridge,
    P4 coefficients, BF16 storage rounding or exact-flush output.
    """
    state = trace['state'].double()
    query = trace['effective_query'].double()
    gradient = trace['gradient'].double()
    if state.ndim != 6 or query.ndim != 6 or gradient.ndim != 6:
        raise ValueError('Expected six-axis window tensors; see README.md')
    L, B, N, H, V, K = state.shape
    W = query.shape[-1]
    if min(L, B, N, H, V, K) < 1 or W < 2:
        raise ValueError('Require nonempty tensors and complete windows with W >= 2')
    omega = basis['omega'].double()
    if omega.ndim != 4 or omega.shape[0] != L or omega.shape[-1] != K:
        raise ValueError('Basis geometry does not match the trace')
    groups, rank = omega.shape[1:3]
    if groups < 1 or H % groups or not 1 <= rank <= K:
        raise ValueError('Invalid group count or basis rank')
    if query.shape != (L, B, N, H, K, W) or gradient.shape != (L, B, N, H, V, W):
        raise ValueError('Query/gradient axes do not match the state')
    if not all(torch.isfinite(x).all() for x in (state, query, gradient, omega)):
        raise ValueError('Non-finite calibration tensors')
    if basis.get('meta', {}).get('coordinates', 'original') != 'original':
        raise ValueError('Basis and traces must use original state coordinates')
    per_head = omega.repeat_interleave(H // groups, dim=1)
    layers = []
    for li in range(L):
        target = state[li] @ query[li]
        U = state[li] @ per_head[li].transpose(-1, -2)
        Q, _ = _mgs_prefix(U, 1e-5)
        output = target.permute(0, 1, 4, 2, 3).contiguous()
        grad = gradient[li].permute(0, 1, 4, 2, 3).contiguous()
        stats = _rank_statistics(grad, output, Q)
        layers.append(stats)
    keys = set.intersection(*(set(x) for x in layers))
    result = {key: torch.stack([x[key] for x in layers]) for key in sorted(keys)}
    result.update(joint_nstep=B * N * W, joint_nseq=B,
                  meta=dict(schema_version=1, coefficient_model='full-gram',
                            omega_granularity='group', exclude_flush=False,
                            storage_rounding_in_score=False, inference_pivots=4,
                            basis_fingerprint=fingerprint(basis['omega'])))
    return result
