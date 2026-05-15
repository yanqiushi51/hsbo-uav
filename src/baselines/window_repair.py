"""
Weak window-aware repair for full-space baseline methods.

This repair only adjusts the timing (start_time) variables toward
task time-window centers. It does NOT use HSBO's hierarchical structure,
lower-layer pre-search, or upper-level surrogate feedback.

The repair is intentionally weak so that repaired baselines remain
distinct from HSBO and serve as fair, interpretable comparisons.
"""

from __future__ import annotations

import numpy as np


def weak_window_repair(x: np.ndarray, env, strength: float = 0.70) -> np.ndarray:
    """Move timing variables toward task time-window centres.

    Parameters
    ----------
    x : ndarray of shape (dim,)
        Full decision vector (upper + lower concatenated).
    env : SparseRewardUAVEnv
    strength : float
        Interpolation weight toward window centre. 0.0 = no repair,
        1.0 = full centring. Default 0.70 is a mild correction.

    Returns
    -------
    x_repaired : ndarray of shape (dim,)
        Vector with timing variables adjusted. Task-key and upper
        variables are unchanged.
    """
    if hasattr(env, "repair_vector"):
        return env.repair_vector(x, strength=strength)

    x = np.asarray(x, dtype=float).copy()
    if x.ndim != 1 or x.shape[0] != env.dim:
        raise ValueError(f"Expected 1-D vector of length {env.dim}, got shape {x.shape}")

    upper = x[: env.dim_upper]
    lower = x[env.dim_upper :]

    # lower layout: [task_key_0, start_norm_0, task_key_1, start_norm_1, ...]
    task_keys = lower[0::2].copy()
    start_norms = lower[1::2].copy()

    n_slots = len(task_keys)  # n * k
    for idx in range(n_slots):
        j = int(min(task_keys[idx] * env.m, env.m - 1))
        window_center = 0.5 * (env.task_earliest[j] + env.task_latest[j])
        center_norm = np.clip(window_center / env.horizon, 0.0, 1.0)
        start_norms[idx] = strength * center_norm + (1.0 - strength) * start_norms[idx]

    lower_repaired = np.empty_like(lower)
    lower_repaired[0::2] = task_keys
    lower_repaired[1::2] = np.clip(start_norms, 0.0, 1.0)

    return np.concatenate([upper, lower_repaired])
