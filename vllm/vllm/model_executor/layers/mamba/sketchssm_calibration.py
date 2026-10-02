# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-head rank allocation and frames of a portable SketchSSM calibration."""

import hashlib
import math
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

FORMAT = "sketchssm-calibration"
SCHEMA_VERSION = 1
# LAPACK QR results depend on the CPU thread count.
EXPORT_THREADS = 2


@contextmanager
def num_threads(n: int):
    previous = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def is_portable_calibration(data: dict[str, Any]) -> bool:
    return data.get("format") == FORMAT


def rank_key(mean_rank: float) -> str:
    return format(float(mean_rank), "g")


def fingerprint(omega: torch.Tensor) -> str:
    """SHA-256 of the basis shape and float32 contents."""
    rows = omega.detach().cpu().float().contiguous()
    h = hashlib.sha256(str(tuple(rows.shape)).encode())
    h.update(rows.numpy().tobytes())
    return h.hexdigest()


def table_digest(m_table: torch.Tensor, dense_table: torch.Tensor) -> str:
    """SHA-256 of the table shape, int16 ranks and bool dense flags."""
    h = hashlib.sha256(str(tuple(m_table.shape)).encode())
    h.update(m_table.to(torch.int16).contiguous().numpy().tobytes())
    h.update(dense_table.to(torch.bool).contiguous().numpy().tobytes())
    return h.hexdigest()


def validate(c: dict[str, Any]) -> None:
    if c.get("format") != FORMAT:
        raise ValueError(f"Not a {FORMAT} file")
    if c.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported calibration schema: {c.get('schema_version')}")
    g, omega = c["geometry"], c["omega"]
    if (
        omega.ndim != 4
        or omega.shape[1] != g["groups"]
        or omega.shape[-1] != g["key_dim"]
    ):
        raise ValueError("Basis shape differs from the declared geometry")
    if fingerprint(omega) != c["basis_fingerprint"]:
        raise ValueError("Basis fingerprint differs from its contents")
    if c["curves"]["meta"].get("basis_fingerprint") != c["basis_fingerprint"]:
        raise ValueError("Score curves were measured for a different basis")
    if c["curves"]["meta"].get("coefficient_model") != "full-gram":
        raise ValueError("Allocation requires Full-Gram error curves")
    error = c["curves"]["joint_dot_sq_sum"]
    if (
        error.ndim != 3
        or error.shape[0] != omega.shape[0]
        or error.shape[1] % g["groups"]
    ):
        raise ValueError("Score curve shape differs from the basis")
    if "layer_ids" in c and len(c["layer_ids"]) != omega.shape[0]:
        raise ValueError("layer_ids length differs from the layer count")


def _solve(objective: np.ndarray, cost: np.ndarray, budget: float) -> np.ndarray:
    """Pick one option per head (the last is dense) under a cost budget."""
    heads, nchoice = objective.shape
    rows = np.arange(heads)
    scaled = objective / max(float(np.mean(np.abs(objective))), 1e-30)

    def lagrangian(lam):
        penalized = scaled + lam * cost[None, :]
        chosen = penalized.argmin(axis=1)
        dual = float(penalized[rows, chosen].sum() - lam * budget)
        return (
            dual,
            float(cost[chosen].sum()),
            float(scaled[rows, chosen].sum()),
            chosen,
        )

    uniform = int(min(nchoice - 1, max(1, np.floor(budget / (heads * cost[0])))))
    best_choice = np.full(heads, uniform - 1, dtype=np.int64)
    if float(cost[best_choice].sum()) > budget + 1e-9:
        raise RuntimeError("failed to construct a feasible presolve incumbent")
    best = float(scaled[rows, best_choice].sum())

    lo, hi = 0.0, 1.0
    while lagrangian(hi)[1] > budget:
        hi *= 2.0
        if not np.isfinite(hi):
            raise RuntimeError("failed to bracket Lagrangian multiplier")
    best_dual, best_lam = -np.inf, hi

    def visit(lam):
        nonlocal best_dual, best_lam, best_choice, best
        dual, used, primal, chosen = lagrangian(lam)
        if dual > best_dual:
            best_dual, best_lam = dual, lam
        if used <= budget and primal < best:
            best_choice, best = chosen.copy(), primal
        return used

    for _ in range(160):
        lam = (lo + hi) / 2.0
        if visit(lam) > budget:
            lo = lam
        else:
            hi = lam
    for lam in (lo, hi, best_lam):
        visit(lam)

    # Greedily spend the remaining budget on the best single-head upgrade.
    choice = best_choice.copy()
    used = float(cost[choice].sum())
    while True:
        delta = cost[None, :] - cost[choice][:, None]
        gain = scaled[rows, choice][:, None] - scaled
        ok = (delta > 0) & (delta <= budget - used + 1e-9) & (gain > 0)
        if not np.any(ok):
            return choice
        head, option = np.unravel_index(
            np.argmax(np.where(ok, gain, -np.inf)), gain.shape
        )
        used += float(delta[head, option])
        choice[head] = option


def crossover(g: dict[str, Any], window: int | None = None) -> int:
    """Dense crossover rank (the default rank cap) of a geometry at a window."""
    K, V = g["key_dim"], g["value_dim"]
    W = g["window"] if window is None else window
    return min(K, K * V // (K + V + (W if g["erase"] else 0)))


def max_rank_at(c: dict[str, Any], window: int | None = None) -> int:
    """Rank cap at a serving window."""
    g = c["geometry"]
    if window is None or window == g["window"]:
        return c["max_rank"]
    cap = crossover(g, window)
    if c["max_rank"] < crossover(g):
        cap = min(cap, c["max_rank"])
    available = c["curves"]["joint_dot_sq_sum"].shape[-1] - 1
    return min(cap, c["omega"].shape[-2], available)


def allocate(
    c: dict[str, Any], mean_rank: float, window: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head ranks ``m_table`` (0 = dense) and ``dense_table``."""
    g = c["geometry"]
    K, V, erase = g["key_dim"], g["value_dim"], g["erase"]
    W = g["window"] if window is None else window
    if any(not isinstance(x, int) or isinstance(x, bool) for x in (K, V, W)):
        raise ValueError("K, V and W must be integers")
    if min(K, V) < 1 or W < 2:
        raise ValueError("Positive state dimensions and W >= 2 are required")
    max_rank = max_rank_at(c, W)
    if not math.isfinite(mean_rank) or mean_rank < 1:
        raise ValueError("Mean-rank budget must be finite and at least 1")
    curves = c["curves"]
    error = torch.as_tensor(curves["joint_dot_sq_sum"]).cpu().double()
    count = int(curves["joint_nstep"])
    if count <= 0:
        raise ValueError("Expected a positive joint_nstep")
    if not torch.isfinite(error).all() or (error < 0).any():
        raise ValueError("Rank scores must be finite and nonnegative")
    error = error / count
    L, H, available = error.shape
    rank_cost = K + V + (W if erase else 0)
    dense_cost = K * V
    cap = min(K, dense_cost // rank_cost)
    if not isinstance(max_rank, int) or isinstance(max_rank, bool):
        raise ValueError("max_rank must be an integer")
    if not 1 <= max_rank <= cap:
        raise ValueError(f"max_rank must lie within [1, {cap}] (the dense crossover)")
    if max_rank >= available or max_rank > c["omega"].shape[-2]:
        raise ValueError("Requested rank cap exceeds the calibrated curves or basis")
    total_budget = rank_cost * mean_rank * L * H
    ranks = np.arange(1, max_rank + 1, dtype=np.int64)
    costs = np.r_[ranks * rank_cost, dense_cost].astype(np.float64)
    objective = torch.cat(
        [
            error[:, :, 1 : max_rank + 1].reshape(-1, max_rank),
            torch.zeros(L * H, 1, dtype=torch.float64),
        ],
        1,
    ).numpy()
    choice = _solve(objective, costs, total_budget)
    dense = choice == max_rank
    m = np.zeros(L * H, dtype=np.int64)
    m[~dense] = ranks[choice[~dense]]
    used = int(m[~dense].sum()) * rank_cost + int(dense.sum()) * dense_cost
    if used > total_budget + 1e-6:
        raise RuntimeError("Recovered allocation exceeds its traffic budget")
    m_table = torch.from_numpy(m).reshape(L, H).to(torch.int16)
    return m_table, torch.from_numpy(dense).reshape(L, H)


def q_full_from(rows: torch.Tensor) -> torch.Tensor:
    """Complete ordered prefix rows to an orthogonal state rotation."""
    state_dim = rows.shape[1]
    augmented = torch.cat(
        [rows.T.double(), torch.eye(state_dim, dtype=torch.float64)], dim=1
    )
    return torch.linalg.qr(augmented)[0].T.contiguous()


def export_frames(omega: torch.Tensor, m_table: torch.Tensor) -> torch.Tensor:
    """Orthogonal frames ``[L, groups, K, K]`` from the basis prefixes."""
    omega = omega.double()
    L, groups, _, K = omega.shape
    m = m_table.long().reshape(L, groups, -1).amax(-1)
    eye = torch.eye(K, dtype=torch.float64)
    frames = torch.empty(L, groups, K, K, dtype=torch.float32)
    for li in range(L):
        for gi in range(groups):
            R = q_full_from(omega[li, gi, : int(m[li, gi])])
            torch.testing.assert_close(R @ R.T, eye, atol=1e-8, rtol=1e-8)
            frames[li, gi] = R.float()
    return frames


def frames_for_mean_rank(
    c: dict[str, Any], mean_rank: float, window: int | None = None
) -> dict[str, Any]:
    """Frames and rank tables of a portable calibration for one mean rank.

    Recorded table digests are checked only at the calibrated allocation.
    """
    with torch.device("cpu"), num_threads(EXPORT_THREADS):
        validate(c)
        mean_rank = float(mean_rank)
        g = c["geometry"]
        W = g["window"] if window is None else window
        m_table, dense_table = allocate(c, mean_rank, W)
        expected = c.get("verified", {}).get(rank_key(mean_rank))
        calibrated = max_rank_at(c, W) == c["max_rank"] and (
            not g["erase"] or g["window"] == W
        )
        unverified_window = expected is not None and not calibrated
        if unverified_window:
            expected = None
        if expected is not None and (
            table_digest(m_table, dense_table) != expected["table_sha256"]
        ):
            raise RuntimeError(
                f"Mean rank {rank_key(mean_rank)} does not reproduce its verified table"
            )
        frames = export_frames(c["omega"], m_table)
    out = dict(
        frames=frames,
        m_table=m_table.long(),
        dense_table=dense_table,
        verified=expected is not None,
        unverified_window=unverified_window,
        window=W,
    )
    if "layer_ids" in c:
        out["layer_ids"] = c["layer_ids"]
    return out
