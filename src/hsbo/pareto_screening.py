from __future__ import annotations

import numpy as np


def pareto_front(scores: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-dominated rows for maximization objectives."""
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # j dominates i if all objectives >= and at least one >
        dominates_i = np.all(scores >= scores[i], axis=1) & np.any(scores > scores[i], axis=1)
        if np.any(dominates_i):
            keep[i] = False
    return keep


def diversity_scores(candidates: np.ndarray, observed: np.ndarray | None) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=float)
    if observed is None or len(observed) == 0:
        return np.ones(candidates.shape[0])
    observed = np.asarray(observed, dtype=float)
    # minimum Euclidean distance to observed points
    d = np.linalg.norm(candidates[:, None, :] - observed[None, :, :], axis=2)
    return d.min(axis=1)


def select_candidates(
    candidates: np.ndarray,
    objectives: np.ndarray,
    observed: np.ndarray | None,
    n_select: int,
    use_pareto: bool = True,
    use_diversity: bool = True,
) -> np.ndarray:
    if len(candidates) == 0:
        return candidates
    objectives = np.asarray(objectives, dtype=float)
    if use_diversity:
        div = diversity_scores(candidates, observed)
        if div.max() > div.min():
            div_norm = (div - div.min()) / (div.max() - div.min() + 1e-12)
        else:
            div_norm = div
        objectives = np.column_stack([objectives, div_norm])
    if use_pareto:
        mask = pareto_front(objectives)
        pool_idx = np.where(mask)[0]
    else:
        scalar = objectives.mean(axis=1)
        pool_idx = np.argsort(-scalar)
    if len(pool_idx) == 0:
        pool_idx = np.arange(len(candidates))
    # Rank the candidate pool by normalized scalar objective.
    obj_pool = objectives[pool_idx]
    obj_norm = (obj_pool - obj_pool.min(axis=0)) / (obj_pool.ptp(axis=0) + 1e-12)
    rank = np.argsort(-obj_norm.mean(axis=1))
    selected = pool_idx[rank[:n_select]]
    return candidates[selected]
