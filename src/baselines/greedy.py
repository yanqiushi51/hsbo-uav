from __future__ import annotations

import numpy as np
from src.evaluation.metrics import ExperimentLogger


def run_greedy_distance(env, budget: int, seed: int, method_name: str = "greedy_distance"):
    """A weak nearest-neighbor route heuristic.

    It does not use exact time-window filtering or synchronization logic. This is
    intended as a fair, transparent baseline rather than an oracle-style decoder.
    """
    rng = np.random.default_rng(seed)
    logger = ExperimentLogger(method_name, seed, env.instance.scale)
    for it in range(1, budget + 1):
        noise = 0.0 if it == 1 else min(0.30, 0.04 + 0.002 * it)
        x = env.policy_vector(rng=rng, policy="distance", noise=noise, allow_window_aware=False)
        result = env.evaluate(x)
        logger.add(it, result)
    return logger.to_dataframe()


def run_greedy_reward_density(env, budget: int, seed: int, method_name: str = "greedy_reward_density"):
    """A reward-density route heuristic.

    It can see task rewards and distances, but it does not solve the continuous
    timing/synchronization subproblem exactly.
    """
    rng = np.random.default_rng(seed)
    logger = ExperimentLogger(method_name, seed, env.instance.scale)
    for it in range(1, budget + 1):
        noise = 0.0 if it == 1 else min(0.30, 0.04 + 0.002 * it)
        x = env.policy_vector(rng=rng, policy="reward_density", noise=noise, allow_window_aware=False)
        result = env.evaluate(x)
        logger.add(it, result)
    return logger.to_dataframe()


def run_greedy(env, budget: int, seed: int, method_name: str = "greedy"):
    # Backward-compatible alias. Use reward-density because it is stronger than
    # nearest-neighbor, but still weaker than the v1 oracle-like greedy vector.
    return run_greedy_reward_density(env, budget=budget, seed=seed, method_name=method_name)
