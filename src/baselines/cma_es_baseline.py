"""
CMA-ES baselines for UAV scheduling.

- run_cmaes               : vanilla CMA-ES in the full continuous space.
- run_cmaes_repair         : CMA-ES with weak window-aware repair.
- run_cmaes_seeded         : CMA-ES initialised from policy_vector.
- run_cmaes_repair_seeded  : CMA-ES + repair, initialised from policy_vector.
"""

from __future__ import annotations

import numpy as np
try:
    import cma
except Exception:  # pragma: no cover - optional dependency
    cma = None

from src.evaluation.metrics import ExperimentLogger
from .window_repair import weak_window_repair


def _init_x0(env, seed: int, use_policy: bool):
    if use_policy:
        rng = np.random.default_rng(seed)
        return env.policy_vector(rng=rng, policy="mixed", noise=0.08)
    return np.random.default_rng(seed).random(env.dim)


def _run_cmaes_core(env, budget: int, seed: int, method_name: str,
                    apply_repair: bool = False, use_policy_seed: bool = False):
    rng = np.random.default_rng(seed)
    logger = ExperimentLogger(method_name, seed, env.instance.scale)

    if cma is None:
        for it in range(1, budget + 1):
            x = env.policy_vector(rng=rng, policy="mixed", noise=0.10) if use_policy_seed and it == 1 else rng.random(env.dim)
            x_eval = weak_window_repair(x, env) if apply_repair else x
            res = env.evaluate(np.clip(x_eval, 0.0, 1.0))
            logger.add(it, res)
        return logger.to_dataframe()

    x0 = _init_x0(env, seed, use_policy_seed)
    sigma0 = 0.25

    opts = {
        "seed": int(seed),
        "bounds": [0.0, 1.0],
        "maxfevals": budget,
        "verbose": -9,
        "CMA_diagonal": True,
        "popsize": min(4 + int(3 * np.log(env.dim)), 32),
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    eval_count = 0

    while eval_count < budget and not es.stop():
        solutions = es.ask()
        pop = []
        for s in solutions:
            if eval_count >= budget:
                break
            x_eval = weak_window_repair(s, env) if apply_repair else np.asarray(s, dtype=float)
            res = env.evaluate(np.clip(x_eval, 0.0, 1.0))
            eval_count += 1
            logger.add(eval_count, res)
            pop.append(-res.reward)

        if len(pop) >= es.sp.weights.mu:
            es.tell(solutions[: len(pop)], pop)

    while eval_count < budget:
        s = rng.random(env.dim)
        x_eval = weak_window_repair(s, env) if apply_repair else s
        res = env.evaluate(np.clip(x_eval, 0.0, 1.0))
        eval_count += 1
        logger.add(eval_count, res)

    return logger.to_dataframe()


def run_cmaes(env, budget: int, seed: int, method_name: str = "cmaes"):
    return _run_cmaes_core(env, budget, seed, method_name, apply_repair=False, use_policy_seed=False)


def run_cmaes_repair(env, budget: int, seed: int, method_name: str = "cmaes_repair"):
    return _run_cmaes_core(env, budget, seed, method_name, apply_repair=True, use_policy_seed=False)


def run_cmaes_seeded(env, budget: int, seed: int, method_name: str = "cmaes_seeded"):
    return _run_cmaes_core(env, budget, seed, method_name, apply_repair=False, use_policy_seed=True)


def run_cmaes_repair_seeded(env, budget: int, seed: int, method_name: str = "cmaes_repair_seeded"):
    return _run_cmaes_core(env, budget, seed, method_name, apply_repair=True, use_policy_seed=True)
