from __future__ import annotations

from .surrogate_bo_core import run_full_space_surrogate_bo


def run_standard_bo(env, budget: int, seed: int, method_name: str = "bo"):
    """Standard GP Bayesian optimization over the full decision vector.

    This is intentionally generic and does not exploit the UAV scheduling
    structure. It is the most important baseline for showing why HSBO is useful.
    """
    return run_full_space_surrogate_bo(env, budget=budget, seed=seed, method_name=method_name, apply_repair=False)
