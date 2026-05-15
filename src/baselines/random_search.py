from __future__ import annotations

import numpy as np
from src.evaluation.metrics import ExperimentLogger


def run_random_search(env, budget: int, seed: int, method_name: str = "random"):
    rng = np.random.default_rng(seed)
    logger = ExperimentLogger(method_name, seed, env.instance.scale)
    for it in range(1, budget + 1):
        x = env.random_vector(rng)
        result = env.evaluate(x)
        logger.add(it, result)
    return logger.to_dataframe()
