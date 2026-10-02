# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Allocate group-basis sketch prefixes independently to each state head."""
import numpy as np
import torch
from .basis import fingerprint
from .traffic import Traffic, AllocationCost
from ._solver import solve


def allocate(curves, basis, *, mean_rank, value_dim, window=16, erase=False, max_rank=None):
    """Use Full-Gram paired scores and the established fixed allocation weights.

    ``max_rank`` caps the sketch rank of a head. It defaults to the dense
    crossover rank ``AllocationCost.crossover`` (a larger rank would cost more
    than the dense fallback); an explicit cap must not exceed it.
    """
    error = torch.as_tensor(curves['joint_dot_sq_sum']).cpu().double()
    count = int(curves['joint_nstep'])
    if error.ndim != 3 or count <= 0:
        raise ValueError('Expected (layers, heads, ranks+1) sums and positive joint_nstep')
    if not torch.isfinite(error).all() or (error < 0).any():
        raise ValueError('Rank scores must be finite and nonnegative')
    error = error / count
    L, H, available = error.shape
    omega = basis['omega'].detach().cpu().float()
    if omega.ndim != 4 or omega.shape[0] != L or omega.shape[1] < 1 or H % omega.shape[1]:
        raise ValueError('Basis groups must divide the state-head count')
    if curves.get('meta', {}).get('coefficient_model') != 'full-gram':
        raise ValueError('Allocation requires Full-Gram error curves')
    identity = fingerprint(omega)
    if curves.get('meta', {}).get('basis_fingerprint') != identity:
        raise ValueError('Scoring basis identity is missing or differs from the supplied basis')
    K = omega.shape[-1]
    traffic = Traffic(K, value_dim, window, erase)
    cost = AllocationCost(K, value_dim, window, erase)
    if max_rank is None:
        max_rank = cost.crossover
    if not isinstance(max_rank, int) or isinstance(max_rank, bool) or not 1 <= max_rank <= cost.crossover:
        raise ValueError(f'max_rank must be an integer within [1, {cost.crossover}] '
                         '(the dense crossover rank, also the default)')
    if max_rank >= available or max_rank > omega.shape[-2]:
        raise ValueError('Requested rank cap exceeds the calibrated curves or basis')
    budget_per_head = cost.budget(mean_rank)
    total_budget = budget_per_head * L * H
    ranks = np.arange(1, max_rank + 1, dtype=np.int64)
    costs = np.r_[ranks * cost.rank, cost.dense].astype(np.float64)
    objective = torch.cat([error[:, :, 1:max_rank + 1].reshape(-1, max_rank),
                           torch.zeros(L * H, 1, dtype=torch.float64)], 1).numpy()
    choice, solver = solve(objective, costs, total_budget)
    dense_np = choice == max_rank
    rank_np = np.zeros(L * H, dtype=np.int64)
    rank_np[~dense_np] = ranks[choice[~dense_np]]
    dense = torch.from_numpy(dense_np).reshape(L, H)
    m = torch.from_numpy(rank_np).reshape(L, H)
    used = int(m[~dense].sum()) * cost.rank + int(dense.sum()) * cost.dense
    if used > total_budget + 1e-6:
        raise RuntimeError('Recovered allocation exceeds its traffic budget')
    mean_nf = (int(m[~dense].sum()) * traffic.sketch_nonflush + int(dense.sum()) * traffic.dense_nonflush) / (L * H)
    report = dict(schema_version=1, inference_pivots=4, exact_coordinates=K,
        omega_granularity='group', rank_granularity='head', mean_rank_budget=mean_rank,
        max_sketch_rank=max_rank, dense_crossover_rank=cost.crossover,
        restricted_candidates=max_rank < cost.crossover, basis_fingerprint=identity,
        score_contract=curves['meta'], traffic=traffic.report(mean_nf),
        allocation_cost=dict(unit='fixed_rank_cost', rank=cost.rank, dense=cost.dense,
                             budget_per_head=budget_per_head, used_total=used),
        dense_heads=int(dense.sum()), sketch_heads=int((~dense).sum()),
        objective=float(np.sum(objective[np.arange(L * H), choice])),
        solver=dict(name='lagrangian', **solver))
    out = dict(m_table=m.to(torch.int16), dense_table=dense, omega=omega, meta=report)
    if 'layer_ids' in basis:
        out['layer_ids'] = basis['layer_ids']
    return out
