from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from src.evaluation.metrics import ExperimentLogger
from .lower_refinement import LowerRefiner, LowerRefinementConfig


@dataclass
class HSBOConfig:
    upper_init: int = 6
    lower_budget: int = 20
    upper_local_noise: float = 0.12
    use_hierarchy: bool = True
    use_presearch: bool = True
    use_gradient: bool = True
    use_pareto: bool = True
    use_diversity: bool = True


class HSBOOptimizer:
    """First research prototype of HSBO.

    The optimizer is intentionally simple and reproducible. It does not claim to
    be the final algorithm; it is a scaffold for validating the paper idea.
    """

    def __init__(self, env, config: HSBOConfig | None = None):
        self.env = env
        self.cfg = config or HSBOConfig()

    def _greedy_upper(self, rng: np.random.Generator, noise: float = 0.0) -> np.ndarray:
        x = self.env.greedy_vector(rng=rng, noise=noise)
        upper, _ = self.env.split(x)
        return upper

    def _sample_upper(self, rng: np.random.Generator, best_upper: np.ndarray | None = None) -> np.ndarray:
        if best_upper is not None and rng.random() < 0.65:
            return np.clip(best_upper + rng.normal(0.0, self.cfg.upper_local_noise, size=self.env.dim_upper), 0.0, 1.0)
        return rng.random(self.env.dim_upper)

    def _policy_presearch_vectors(self, rng: np.random.Generator) -> list[np.ndarray]:
        if not self.cfg.use_presearch or not hasattr(self.env, "policy_vector"):
            return []
        if hasattr(self.env, "rate_threshold_mbps"):
            policies = ["max_rate", "edf_rate", "nearest_feasible", "reward_density", "mixed"]
        else:
            policies = ["reward_density", "distance", "mixed"]
        vectors = []
        for policy in policies:
            for noise in (0.0, 0.04, 0.09):
                try:
                    vectors.append(
                        self.env.policy_vector(
                            rng=rng,
                            policy=policy,
                            noise=noise,
                            allow_window_aware=True,
                        )
                    )
                except Exception:
                    continue
        return vectors

    def run(self, budget: int, seed: int, method_name: str = "hsbo"):
        rng = np.random.default_rng(seed)
        logger = ExperimentLogger(method_name, seed, self.env.instance.scale)

        if not self.cfg.use_hierarchy:
            # Ablation: remove hierarchy by performing random full-space BO-like local search
            # around greedy vectors. This is deliberately weaker but fair under same budget.
            best_x = self.env.greedy_vector(rng, noise=0.0)
            best_res = None
            for it in range(1, budget + 1):
                if it == 1:
                    x = best_x
                elif best_res is not None and rng.random() < 0.55:
                    x = np.clip(best_x + rng.normal(0.0, 0.12, size=self.env.dim), 0.0, 1.0)
                else:
                    x = self.env.random_vector(rng)
                res = self.env.evaluate(x)
                if best_res is None or res.reward > best_res.reward:
                    best_res = res
                    best_x = x.copy()
                logger.add(it, res)
            return logger.to_dataframe()

        lower_cfg = LowerRefinementConfig(
            init_points=min(18, max(8, self.cfg.lower_budget // 2)),
            candidate_pool=140,
            batch_size=4,
            local_noise=0.10,
            use_presearch=self.cfg.use_presearch,
            use_gradient=self.cfg.use_gradient,
            use_pareto=self.cfg.use_pareto,
            use_diversity=self.cfg.use_diversity,
        )
        refiner = LowerRefiner(self.env, lower_cfg)

        eval_used = 0
        upper_records = []
        best_upper = None
        best_result = None

        for x in self._policy_presearch_vectors(rng):
            if eval_used >= budget:
                break
            res = self.env.evaluate(x)
            eval_used += 1
            logger.add(eval_used, res)
            upper, _ = self.env.split(x)
            upper_records.append((upper.copy(), res.reward))
            if best_result is None or res.reward > best_result.reward:
                best_result = res
                best_upper = upper.copy()

        # Initial upper candidates include a greedy flight strategy and random variants.
        initial_uppers = [self._greedy_upper(rng, noise=0.0)]
        for _ in range(self.cfg.upper_init - 1):
            if rng.random() < 0.45:
                initial_uppers.append(self._greedy_upper(rng, noise=rng.uniform(0.03, 0.15)))
            else:
                initial_uppers.append(rng.random(self.env.dim_upper))

        upper_queue = initial_uppers
        while eval_used < budget:
            if upper_queue:
                upper = upper_queue.pop(0)
            else:
                upper = self._sample_upper(rng, best_upper)

            chunk = min(self.cfg.lower_budget, budget - eval_used)
            _, res, consumed = refiner.refine(
                upper=upper,
                budget=chunk,
                rng=rng,
                start_iteration=eval_used,
                logger=logger,
            )
            eval_used += consumed
            if res is not None:
                upper_records.append((upper.copy(), res.reward))
                if best_result is None or res.reward > best_result.reward:
                    best_result = res
                    best_upper = upper.copy()

            # Add a couple of upper perturbations around the best upper to emulate upper-level surrogate screening.
            if best_upper is not None and len(upper_queue) < 2 and eval_used < budget:
                upper_queue.append(self._sample_upper(rng, best_upper))
                if rng.random() < 0.35:
                    upper_queue.append(rng.random(self.env.dim_upper))
        return logger.to_dataframe()


def run_hsbo(env, budget: int, seed: int, method_name: str = "hsbo", **kwargs):
    cfg = HSBOConfig(**kwargs)
    return HSBOOptimizer(env, cfg).run(budget=budget, seed=seed, method_name=method_name)
