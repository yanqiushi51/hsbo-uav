from __future__ import annotations

import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import ExperimentLogger
from .window_repair import weak_window_repair


def _fit_gp(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()
    kernel = ConstantKernel(1.0, (0.1, 10.0)) * Matern(
        length_scale=np.ones(X.shape[1]),
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-6, 1e-2))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=False,
        n_restarts_optimizer=0,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X, y_scaled)
    return gp


def _training_subset(X: np.ndarray, y: np.ndarray, max_points: int = 90) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= max_points:
        return X, y
    n_top = max_points // 2
    top = np.argsort(-y)[:n_top]
    recent = np.arange(max(0, len(X) - (max_points - n_top)), len(X))
    idx = np.unique(np.concatenate([top, recent]))
    if len(idx) > max_points:
        idx = idx[-max_points:]
    return X[idx], y[idx]


def _candidate_pool(rng: np.random.Generator, env, best_x: np.ndarray, n: int = 640) -> np.ndarray:
    d = env.dim
    local_count = n // 2
    local = np.clip(best_x + rng.normal(0.0, 0.12, size=(local_count, d)), 0.0, 1.0)
    random = rng.random((n - local_count, d))
    return np.vstack([local, random])


def _lhs(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    cut = np.linspace(0.0, 1.0, n + 1)
    u = rng.random((n, d))
    points = cut[:-1, None] + u * (cut[1:] - cut[:-1])[:, None]
    for j in range(d):
        rng.shuffle(points[:, j])
    return np.clip(points, 0.0, 1.0)


def _repair_many(candidates: np.ndarray, env, strength: float = 0.90) -> np.ndarray:
    return np.asarray([weak_window_repair(x, env, strength=strength) for x in candidates], dtype=float)


def run_full_space_surrogate_bo(
    env,
    budget: int,
    seed: int,
    method_name: str,
    apply_repair: bool = False,
):
    """Practical full-space GP surrogate BO baseline.

    The surrogate is fit directly in the complete decision space. To keep
    repeated 40-D/80-D baselines executable, each refit uses the strongest and
    most recent observations rather than an ever-growing full GP matrix.
    """
    rng = np.random.default_rng(seed)
    logger = ExperimentLogger(method_name, seed, env.instance.scale)
    init_points = min(24, max(10, env.dim // 2))
    X = []
    y = []
    best_x = None
    best_y = -np.inf
    initial = _lhs(rng, init_points, env.dim)

    for it in range(1, budget + 1):
        if it <= init_points:
            x = initial[it - 1]
        elif len(X) < 6 or np.std(y) < 1e-9:
            x = rng.random(env.dim)
        else:
            X_arr = np.asarray(X, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            X_fit, y_fit = _training_subset(X_arr, y_arr)
            try:
                gp = _fit_gp(X_fit, y_fit)
                candidates = _candidate_pool(rng, env, best_x)
                mu, std = gp.predict(candidates, return_std=True)
                acq = mu + 0.8 * std
                x = candidates[int(np.argmax(acq))]
            except Exception:
                x = np.clip(best_x + rng.normal(0.0, 0.15, size=env.dim), 0.0, 1.0)

        x_eval = weak_window_repair(x, env) if apply_repair else np.asarray(x, dtype=float)
        res = env.evaluate(np.clip(x_eval, 0.0, 1.0))
        reward = float(res.reward)
        X.append(np.clip(x, 0.0, 1.0))
        y.append(reward)
        if reward > best_y or best_x is None:
            best_y = reward
            best_x = np.clip(x, 0.0, 1.0).copy()
        logger.add(it, res)

    return logger.to_dataframe()


def run_full_space_refinement_bo(
    env,
    budget: int,
    seed: int,
    method_name: str,
):
    """Clean no-hierarchy variant: full-space surrogate over refined schedules.

    This removes upper/lower decomposition and the upper mobility surrogate.
    Every candidate is first link-window/rate refined, then the GP is trained
    directly on the refined complete decision vector.
    """
    rng = np.random.default_rng(seed)
    logger = ExperimentLogger(method_name, seed, env.instance.scale)
    init_points = min(24, max(12, env.dim // 3))
    X = []
    y = []
    best_x = None
    best_y = -np.inf

    initial = _lhs(rng, init_points, env.dim)
    for it in range(1, budget + 1):
        if it <= init_points:
            x_refined = weak_window_repair(initial[it - 1], env, strength=0.90)
        elif len(X) < 6 or np.std(y) < 1e-9 or best_x is None:
            x_refined = weak_window_repair(rng.random(env.dim), env, strength=0.90)
        else:
            X_arr = np.asarray(X, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            X_fit, y_fit = _training_subset(X_arr, y_arr, max_points=70)
            try:
                gp = _fit_gp(X_fit, y_fit)
                raw_pool = _candidate_pool(rng, env, best_x, n=192)
                mu_raw, std_raw = gp.predict(raw_pool, return_std=True)
                raw_acq = mu_raw + 0.8 * std_raw
                shortlist = np.argsort(-raw_acq)[:16]
                refined_pool = _repair_many(raw_pool[shortlist], env, strength=0.90)
                mu, std = gp.predict(refined_pool, return_std=True)
                acq = mu + 0.8 * std
                x_refined = refined_pool[int(np.argmax(acq))]
            except Exception:
                x_refined = weak_window_repair(
                    np.clip(best_x + rng.normal(0.0, 0.15, size=env.dim), 0.0, 1.0),
                    env,
                    strength=0.90,
                )

        x_refined = np.clip(x_refined, 0.0, 1.0)
        res = env.evaluate(x_refined)
        reward = float(res.reward)
        X.append(x_refined.copy())
        y.append(reward)
        if reward > best_y or best_x is None:
            best_y = reward
            best_x = x_refined.copy()
        logger.add(it, res)

    return logger.to_dataframe()
