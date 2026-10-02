# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Preserve an existing group basis and rank/dense tables without reallocation."""
import torch
from .basis import fingerprint
from .traffic import Traffic


def reuse(allocation, *, value_dim, window=16, erase=False):
    omega = allocation['omega'].detach().cpu()
    ranks = allocation['m_table'].detach().cpu()
    dense = allocation['dense_table'].detach().cpu()
    if omega.ndim != 4 or ranks.ndim != 2 or ranks.shape != dense.shape:
        raise ValueError('Expected a group basis and matching per-head rank/dense tables')
    L, groups, available, K = omega.shape
    if groups < 1 or ranks.shape[0] != L or ranks.shape[1] < 1 or ranks.shape[1] % groups:
        raise ValueError('Invalid group-to-head geometry')
    if not torch.isfinite(omega).all() or not torch.isfinite(ranks).all():
        raise ValueError('Non-finite allocation')
    if not torch.equal(ranks, ranks.long()) or dense.dtype != torch.bool:
        raise ValueError('Ranks must be integers and dense flags must be boolean')
    if (ranks < 0).any() or (ranks > available).any() or not torch.equal(ranks == 0, dense):
        raise ValueError('Invalid rank or dense sentinel')
    traffic = Traffic(K, value_dim, window, erase)
    nf = torch.where(dense, float(traffic.dense_nonflush), ranks.double() * traffic.sketch_nonflush).mean().item()
    source = allocation.get('meta', {})
    # Keep numerical provenance, not source-machine filenames or arbitrary objects.
    keys = ('mean_rank_budget', 'budget_per_head_including_flush', 'flush_per_head',
            'latch_cost', 'dense_cost', 'm_star', 'max_sketch_rank')
    cost = {k: source[k] for k in keys if isinstance(source.get(k), (int, float))}
    out = dict(omega=omega.clone(), m_table=ranks.clone(), dense_table=dense.clone(),
                meta=dict(schema_version=1, allocation_policy='reuse_unchanged',
                          basis_fingerprint=fingerprint(omega), source_cost=cost,
                          dense_heads=int(dense.sum()), sketch_heads=int((~dense).sum()),
                          inference_pivots=4, exact_coordinates=K,
                          omega_granularity='group', rank_granularity='head',
                          traffic=traffic.report(nf)))
    if 'layer_ids' in allocation:
        out['layer_ids'] = allocation['layer_ids']
    return out
