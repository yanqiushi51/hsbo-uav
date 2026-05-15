from __future__ import annotations

import numpy as np

from src.evaluation.metrics import ExperimentLogger
from .surrogate_bo_core import run_full_space_surrogate_bo


def _run_policy(env, budget: int, seed: int, method_name: str, policy: str):
    rng = np.random.default_rng(seed)
    logger = ExperimentLogger(method_name, seed, env.instance.scale)
    for it in range(1, budget + 1):
        noise = 0.0 if it == 1 else min(0.25, 0.025 + 0.0015 * it)
        x = env.policy_vector(rng=rng, policy=policy, noise=noise, allow_window_aware=True)
        res = env.evaluate(x)
        logger.add(it, res)
    return logger.to_dataframe()


def run_max_rate_greedy(env, budget: int, seed: int, method_name: str = "max_rate_greedy"):
    return _run_policy(env, budget, seed, method_name, policy="max_rate")


def run_edf_rate_greedy(env, budget: int, seed: int, method_name: str = "edf_rate_greedy"):
    return _run_policy(env, budget, seed, method_name, policy="edf_rate")


def run_nearest_feasible_user(env, budget: int, seed: int, method_name: str = "nearest_feasible_user"):
    return _run_policy(env, budget, seed, method_name, policy="nearest_feasible")


def run_rate_window_repair_bo(env, budget: int, seed: int, method_name: str = "rate_window_repair_bo"):
    return run_full_space_surrogate_bo(env, budget=budget, seed=seed, method_name=method_name, apply_repair=True)
