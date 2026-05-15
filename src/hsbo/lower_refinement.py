from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
import warnings

from .pareto_screening import select_candidates


@dataclass
class LowerRefinementConfig:
    init_points: int = 18
    candidate_pool: int = 160
    batch_size: int = 4
    local_noise: float = 0.10
    use_presearch: bool = True
    use_gradient: bool = True
    use_pareto: bool = True
    use_diversity: bool = True


class LowerRefiner:
    def __init__(self, env, config: LowerRefinementConfig | None = None):
        self.env = env
        self.cfg = config or LowerRefinementConfig()

    def _make_full(self, upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
        return np.concatenate([np.clip(upper, 0, 1), np.clip(lower, 0, 1)])

    def _heuristic_lower(self, rng: np.random.Generator, noise: float = 0.0) -> np.ndarray:
        xg = self.env.greedy_vector(rng=rng, noise=noise)
        _, lower = self.env.split(xg)
        return lower

    def _initial_lowers(self, rng: np.random.Generator) -> list[np.ndarray]:
        lowers = []
        if self.cfg.use_presearch:
            lowers.append(self._heuristic_lower(rng, noise=0.0))
            for _ in range(max(1, self.cfg.init_points // 3)):
                lowers.append(self._heuristic_lower(rng, noise=rng.uniform(0.03, 0.18)))
        while len(lowers) < self.cfg.init_points:
            lowers.append(rng.random(self.env.dim_lower))
        return lowers[: self.cfg.init_points]

    def _fit_surrogate(self, X: np.ndarray, y: np.ndarray):
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()
        kernel = ConstantKernel(1.0, (0.1, 10.0)) * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-3, 1e3), nu=2.5) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-6, 1e-2))
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
        return gp, scaler

    def _surrogate_gradient_norm(self, gp, X: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        # Cheap finite-difference on surrogate mean, not expensive simulator.
        if not self.cfg.use_gradient:
            return np.zeros(X.shape[0])
        n, d = X.shape
        # For speed, sample a subset of dimensions for high-dimensional lower spaces.
        max_dims = min(d, 8)
        dims = np.linspace(0, d - 1, max_dims, dtype=int)
        base = gp.predict(X, return_std=False)
        grad_sq = np.zeros(n)
        for j in dims:
            xp = X.copy()
            xm = X.copy()
            xp[:, j] = np.clip(xp[:, j] + eps, 0.0, 1.0)
            xm[:, j] = np.clip(xm[:, j] - eps, 0.0, 1.0)
            diff = (gp.predict(xp, return_std=False) - gp.predict(xm, return_std=False)) / (2.0 * eps)
            grad_sq += diff * diff
        return np.sqrt(grad_sq / max_dims)

    def _candidate_pool(self, rng: np.random.Generator, best_lower: np.ndarray, X_obs: np.ndarray) -> np.ndarray:
        n = self.cfg.candidate_pool
        d = self.env.dim_lower
        n_local = n // 2
        local = np.clip(best_lower + rng.normal(0.0, self.cfg.local_noise, size=(n_local, d)), 0.0, 1.0)
        random = rng.random((n - n_local, d))
        if self.cfg.use_presearch:
            h = np.array([self._heuristic_lower(rng, noise=rng.uniform(0.02, 0.20)) for _ in range(max(2, n // 12))])
            pool = np.vstack([local, random, h])
        else:
            pool = np.vstack([local, random])
        return np.clip(pool, 0.0, 1.0)

    def refine(self, upper: np.ndarray, budget: int, rng: np.random.Generator, start_iteration: int, logger=None):
        X = []
        y = []
        best_lower = None
        best_result = None
        evals = 0

        # Initial lower-level observations.
        for lower in self._initial_lowers(rng):
            if evals >= budget:
                break
            result = self.env.evaluate(self._make_full(upper, lower))
            evals += 1
            if logger is not None:
                logger.add(start_iteration + evals, result)
            X.append(lower)
            y.append(result.reward)
            if best_result is None or result.reward > best_result.reward:
                best_result = result
                best_lower = lower.copy()

        # Surrogate-guided lower-level refinement.
        while evals < budget:
            X_arr = np.asarray(X)
            y_arr = np.asarray(y)
            if len(X_arr) < 5 or np.std(y_arr) < 1e-9:
                candidates = self._candidate_pool(rng, best_lower, X_arr)
                selected = candidates[: min(self.cfg.batch_size, budget - evals)]
            else:
                try:
                    gp, scaler = self._fit_surrogate(X_arr, y_arr)
                    candidates = self._candidate_pool(rng, best_lower, X_arr)
                    mu, std = gp.predict(candidates, return_std=True)
                    grad = self._surrogate_gradient_norm(gp, candidates)
                    # Objectives are maximized: predicted value, uncertainty, gradient-drive.
                    objectives = np.column_stack([mu, std, grad])
                    selected = select_candidates(
                        candidates,
                        objectives,
                        observed=X_arr,
                        n_select=min(self.cfg.batch_size, budget - evals),
                        use_pareto=self.cfg.use_pareto,
                        use_diversity=self.cfg.use_diversity,
                    )
                except Exception:
                    candidates = self._candidate_pool(rng, best_lower, X_arr)
                    selected = candidates[: min(self.cfg.batch_size, budget - evals)]
            for lower in selected:
                if evals >= budget:
                    break
                result = self.env.evaluate(self._make_full(upper, lower))
                evals += 1
                if logger is not None:
                    logger.add(start_iteration + evals, result)
                X.append(lower)
                y.append(result.reward)
                if best_result is None or result.reward > best_result.reward:
                    best_result = result
                    best_lower = lower.copy()
        return best_lower, best_result, evals
