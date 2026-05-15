"""
Differential Evolution baselines for UAV scheduling.

- run_de                : vanilla DE/rand/1/bin in the full continuous space.
- run_de_repair         : DE with weak window-aware repair.
- run_de_seeded         : DE with first individual from policy_vector.
- run_de_repair_seeded  : DE + repair with first individual from policy_vector.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import ExperimentLogger
from .window_repair import weak_window_repair


def _run_de_core(env, budget: int, seed: int, method_name: str,
                 apply_repair: bool = False, use_policy_seed: bool = False):
    rng = np.random.default_rng(seed)
    logger = ExperimentLogger(method_name, seed, env.instance.scale)

    pop_size = min(60, max(20, 5 * env.dim))
    pop = rng.random((pop_size, env.dim))

    if use_policy_seed:
        pop[0] = env.policy_vector(rng=rng, policy="mixed", noise=0.08)

    fitness = np.full(pop_size, -np.inf)
    eval_count = 0

    F = 0.75
    CR = 0.80

    def evaluate(idx: int, x: np.ndarray):
        nonlocal eval_count
        res = env.evaluate(np.clip(x, 0.0, 1.0))
        eval_count += 1
        logger.add(eval_count, res)
        return res.reward, res

    for i in range(pop_size):
        if eval_count >= budget:
            break
        x_eval = weak_window_repair(pop[i], env) if apply_repair else pop[i]
        reward, res = evaluate(i, x_eval)
        fitness[i] = reward

    while eval_count < budget:
        for i in range(pop_size):
            if eval_count >= budget:
                break

            candidates = [j for j in range(pop_size) if j != i]
            a, b, c = pop[rng.choice(candidates, size=3, replace=False)]
            mutant = a + F * (b - c)

            cross_mask = rng.random(env.dim) < CR
            if not cross_mask.any():
                cross_mask[rng.integers(0, env.dim)] = True
            trial = np.where(cross_mask, mutant, pop[i])
            trial = np.clip(trial, 0.0, 1.0)

            x_eval = weak_window_repair(trial, env) if apply_repair else trial
            reward, res = evaluate(i, x_eval)

            if reward > fitness[i]:
                pop[i] = trial.copy()
                fitness[i] = reward

    return logger.to_dataframe()


def run_de(env, budget: int, seed: int, method_name: str = "de"):
    return _run_de_core(env, budget, seed, method_name, apply_repair=False, use_policy_seed=False)


def run_de_repair(env, budget: int, seed: int, method_name: str = "de_repair"):
    return _run_de_core(env, budget, seed, method_name, apply_repair=True, use_policy_seed=False)


def run_de_seeded(env, budget: int, seed: int, method_name: str = "de_seeded"):
    return _run_de_core(env, budget, seed, method_name, apply_repair=False, use_policy_seed=True)


def run_de_repair_seeded(env, budget: int, seed: int, method_name: str = "de_repair_seeded"):
    return _run_de_core(env, budget, seed, method_name, apply_repair=True, use_policy_seed=True)
