# Project Structure

The repository is organized for the paper-writing stage. The top level now
shows only the active project areas:

- `paper/`
- `archive/`
- `src/`
- `scripts/`
- `docs/`
- dependency and Git metadata files

## Paper

- `paper/main.tex`: active manuscript source.
- `paper/figures/`: figures referenced by `paper/main.tex`.
- `paper/data/export/`: selected CSV exports used to draw paper figures.
- `paper/figure_scripts/`: figure-generation scripts and legacy figure helpers.
- `paper/package/`: packaged manuscript snapshots.
- `paper/deliverables/`: selected tables and zipped deliverable bundles.
- `paper/qa/`: visual checks and page snapshots.
- `paper/notes/`: extracted manuscript text and other writing notes.

Compile the manuscript from inside `paper/` so paths like `figures/fig1.pdf`
resolve correctly.

## Archive

- `archive/data/`: committed dataset and calibration snapshots.
- `archive/experiments/`: committed result tables, archived logs, smoke runs,
  calibration summaries, and diagnostic figures.

The archive is intentionally read-mostly. It preserves the completed experiment
record without cluttering the root of the writing repository.

## Code

- `src/hsbo/`: hierarchical surrogate Bayesian optimization implementation.
- `src/problem/`: sparse-reward UAV and wireless service environments.
- `src/baselines/`: random, greedy, BO, GA, PSO, DE, CMA-ES, repair, and wireless
  heuristic baselines.
- `src/evaluation/`: logging and summary utilities.
- `scripts/data/`: dataset generation entry points.
- `scripts/experiments/`: optional experiment rerun entry points.
- `scripts/summaries/`: summary-table builders.
- `scripts/figures/`: figure-generation and legacy plotting helpers.
- `scripts/maintenance/`: cleanup and repository maintenance utilities.

## Local Generated Files

- `_local/`: ignored bucket for moved local build files and runtime logs.
- `outputs/`, `runs/`, `scratch/`, `local_results/`, `local_logs/`: ignored
  locations for any future reruns.

Use `python scripts/maintenance/clean_generated.py --dry-run` to preview ignored generated
files. Use `--apply` only after checking the list.

## Policy

1. Put writing work in `paper/`.
2. Keep completed experimental evidence in `archive/`.
3. Put new exploratory outputs in `outputs/` or `runs/`, not in the repository
   root.
4. Keep source changes in `src/` and `scripts/`.
5. Do not delete archived results unless the paper no longer depends on them.
