from __future__ import annotations

import numpy as np
from src.evaluation.metrics import ExperimentLogger


def run_pso(env, budget: int, seed: int, method_name: str = "pso", swarm_size: int | None = None):
    rng = np.random.default_rng(seed)
    swarm_size = swarm_size or min(48, max(16, env.dim * 2))
    logger = ExperimentLogger(method_name, seed, env.instance.scale)

    x = rng.random((swarm_size, env.dim))
    v = rng.normal(0.0, 0.08, size=x.shape)
    pbest = x.copy()
    pbest_score = np.full(swarm_size, -np.inf)
    gbest = x[0].copy()
    gbest_score = -np.inf
    eval_count = 0

    while eval_count < budget:
        for i in range(swarm_size):
            if eval_count >= budget:
                break
            res = env.evaluate(x[i])
            eval_count += 1
            if res.reward > pbest_score[i]:
                pbest_score[i] = res.reward
                pbest[i] = x[i].copy()
            if res.reward > gbest_score:
                gbest_score = res.reward
                gbest = x[i].copy()
            logger.add(eval_count, res)
        # Update particles.
        w = 0.72
        c1 = 1.45
        c2 = 1.45
        r1 = rng.random(x.shape)
        r2 = rng.random(x.shape)
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        v = np.clip(v, -0.25, 0.25)
        x = np.clip(x + v, 0.0, 1.0)
    return logger.to_dataframe()
