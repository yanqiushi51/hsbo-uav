from __future__ import annotations

import numpy as np
from src.evaluation.metrics import ExperimentLogger


def run_ga(env, budget: int, seed: int, method_name: str = "ga", pop_size: int | None = None):
    rng = np.random.default_rng(seed)
    pop_size = pop_size or min(48, max(16, env.dim * 2))
    logger = ExperimentLogger(method_name, seed, env.instance.scale)

    population = rng.random((pop_size, env.dim))
    fitness = np.zeros(pop_size, dtype=float)
    eval_count = 0

    def evaluate_individual(idx: int):
        nonlocal eval_count
        res = env.evaluate(population[idx])
        eval_count += 1
        fitness[idx] = res.reward
        logger.add(eval_count, res)

    for i in range(pop_size):
        if eval_count >= budget:
            break
        evaluate_individual(i)

    while eval_count < budget:
        # Tournament selection.
        new_pop = []
        elite_idx = np.argsort(-fitness)[: max(2, pop_size // 8)]
        new_pop.extend(population[elite_idx])
        while len(new_pop) < pop_size:
            cand = rng.integers(0, pop_size, size=3)
            p1 = population[cand[np.argmax(fitness[cand])]]
            cand = rng.integers(0, pop_size, size=3)
            p2 = population[cand[np.argmax(fitness[cand])]]
            mask = rng.random(env.dim) < 0.5
            child = np.where(mask, p1, p2)
            # Gaussian mutation plus occasional reset mutation.
            mut_mask = rng.random(env.dim) < 0.12
            child = child.copy()
            child[mut_mask] += rng.normal(0.0, 0.10, size=mut_mask.sum())
            reset_mask = rng.random(env.dim) < 0.02
            child[reset_mask] = rng.random(reset_mask.sum())
            new_pop.append(np.clip(child, 0.0, 1.0))
        population = np.asarray(new_pop[:pop_size])
        for i in range(pop_size):
            if eval_count >= budget:
                break
            evaluate_individual(i)
    return logger.to_dataframe()
