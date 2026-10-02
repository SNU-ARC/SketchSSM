# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Deterministic Lagrangian allocation with a reported lower bound and gap."""
import numpy as np


def solve(objective, cost, budget):
    """Select one rank/dense option per head; the last option is dense.

    This is a feasible bounded-gap solution, not a claim of exact MILP optimality.
    Costs are rank-linear followed by one full-state cost.
    """
    heads, nchoice = objective.shape
    objective_scale = max(float(np.mean(np.abs(objective))), 1e-30)
    scaled_objective = objective / objective_scale
    def lagrangian(lam):
        penalized = scaled_objective + lam * cost[None, :]
        chosen = penalized.argmin(axis=1)
        row_min = penalized[np.arange(heads), chosen]
        used_lag = float(cost[chosen].sum())
        dual = float(row_min.sum() - lam * budget)
        primal = float(
            scaled_objective[np.arange(heads), chosen].sum())
        return dual, used_lag, primal, chosen, row_min

    uniform_rank = int(min(
        nchoice - 1,
        max(1,
            np.floor(budget / (heads * cost[0])))))
    uniform_choice = np.full(
        heads, uniform_rank - 1, dtype=np.int64)
    uniform_used = float(cost[uniform_choice].sum())
    if uniform_used > budget + 1e-9:
        raise RuntimeError("failed to construct a feasible presolve incumbent")
    incumbent_choice = uniform_choice
    incumbent_objective = float(
        scaled_objective[np.arange(heads), uniform_choice].sum())

    lo, hi = 0.0, 1.0
    while lagrangian(hi)[1] > budget:
        hi *= 2.0
        if not np.isfinite(hi):
            raise RuntimeError("failed to bracket Lagrangian multiplier")
    best_dual = -np.inf
    best_lam = hi
    for _ in range(160):
        lam = (lo + hi) / 2.0
        dual, used_lag, primal, chosen, _ = lagrangian(lam)
        if dual > best_dual:
            best_dual, best_lam = dual, lam
        if used_lag <= budget and primal < incumbent_objective:
            incumbent_choice = chosen.copy()
            incumbent_objective = primal
        if used_lag > budget:
            lo = lam
        else:
            hi = lam
    for lam in (lo, hi, best_lam):
        dual, used_lag, primal, chosen, _ = lagrangian(lam)
        if dual > best_dual:
            best_dual, best_lam = dual, lam
        if used_lag <= budget and primal < incumbent_objective:
            incumbent_choice = chosen.copy()
            incumbent_objective = primal

    safety = 1e-8 * max(1.0, abs(best_dual), abs(incumbent_objective))
    certified_lower = best_dual - safety
    choice = incumbent_choice.copy()
    used_choice = float(cost[choice].sum())

    local_updates = 0
    while True:
        slack = budget - used_choice
        current_cost = cost[choice]
        current_obj = scaled_objective[np.arange(heads), choice]
        delta_cost = cost[None, :] - current_cost[:, None]
        improvement = current_obj[:, None] - scaled_objective
        feasible = (delta_cost > 0) & (delta_cost <= slack + 1e-9) \
            & (improvement > 0)
        if not np.any(feasible):
            break
        candidate_improvement = np.where(feasible, improvement, -np.inf)
        head_idx, choice_idx = np.unravel_index(
            np.argmax(candidate_improvement), candidate_improvement.shape)
        used_choice += float(delta_cost[head_idx, choice_idx])
        choice[head_idx] = choice_idx
        local_updates += 1
    incumbent_objective = float(
        scaled_objective[np.arange(heads), choice].sum())
    certified_gap = max(0.0, incumbent_objective - certified_lower)
    solver_gap = float(certified_gap / max(abs(incumbent_objective), 1e-30))

    return choice, {"relative_gap": solver_gap,
                    "lower_bound": certified_lower * objective_scale,
                    "objective": incumbent_objective * objective_scale,
                    "local_fill_updates": local_updates}
