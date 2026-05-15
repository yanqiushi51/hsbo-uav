# Source Code Map

The source package is already split by responsibility. Keep implementation
changes inside these module boundaries.

## `src/hsbo/`

Core FT-HSBO optimization logic:

- `hsbo_full.py`: main optimizer and public `run_hsbo` entry point.
- `lower_refinement.py`: lower-level refinement around candidate schedules.
- `pareto_screening.py`: Pareto filtering and diversity-aware candidate
  selection.

## `src/problem/`

Problem definitions and instance generators:

- `sparse_reward_env.py`: original sparse-reward cooperative scheduling
  evaluator.
- `benchmark_generator.py`: sparse scheduling benchmark generator/loader.
- `wireless_service_env.py`: wireless WCL service evaluator.
- `wireless_generator.py`: wireless WCL instance generator/loader.

## `src/baselines/`

Baseline optimizers and heuristics:

- `random_search.py`, `greedy.py`, `ga.py`, `pso.py`, `standard_bo.py`
- `de_baseline.py`, `cma_es_baseline.py`
- `surrogate_bo_core.py`, `window_repair.py`
- `wireless_heuristics.py`

## `src/evaluation/`

Shared experiment logging and summary utilities.

## Import Policy

Scripts insert the repository root into `sys.path` and import modules through
the `src.` package path. Core source files should continue using explicit
`src.*` imports rather than relative imports so script entry points remain
simple.
