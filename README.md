# HSBO-UAV v2: Sparse-Reward Cooperative Multi-UAV Scheduling

This is the second runnable research-code scaffold for:

**Hierarchical Surrogate Bayesian Optimization for Sparse-Reward Multi-UAV Mission Scheduling**

Compared with v1, this version fixes the main benchmark flaw: the previous greedy seed could exploit the task-key encoding too directly and became too strong. Version 2 makes the benchmark closer to a real sparse-reward cooperative scheduling problem.

## What changed in v2

1. **No oracle-style greedy baseline**
   - `greedy_vector()` is no longer the main baseline logic.
   - New baselines:
     - `greedy_distance`: nearest-task route heuristic.
     - `greedy_reward_density`: reward/travel proxy heuristic.
   - These heuristics do not solve the continuous timing or cooperative synchronization subproblem exactly.

2. **True scheduled-time semantics**
   - The lower-level `start_time` variable is now treated as the actual scheduled service start.
   - A task scores only if the UAV can arrive by the scheduled start and the start lies inside the task time window.
   - This makes random timing sparse and prevents route-only heuristics from becoming near-oracle solvers.

3. **Cooperative sparse-reward tasks**
   - Some high-value tasks require two or more distinct UAVs to arrive within a synchronization tolerance.
   - A single UAV receives no reward for a cooperative task.
   - This is the safe, publishable abstraction of the original multi-agent synchronized countermeasure idea.

4. **Narrower time windows**
   - `small`, `medium`, and `large` instances now use increasingly narrow time windows.
   - This creates a stronger high-dimensional sparse-reward setting.

5. **Cleaner metrics**
   - Logs now include `best_served_tasks`.
   - Summary reports `final_best_served_mean`, which corresponds to the best-so-far solution rather than the last sampled solution.

## Installation

```bash
conda create -n hsbo-uav python=3.10 -y
conda activate hsbo-uav
pip install -r requirements.txt
```

`scikit-optimize` is optional but recommended. If unavailable, `bo` falls back to random search and prints a warning.

## Writing-stage repository layout

- `paper/`: manuscript, paper figures, figure data exports, paper packages, and visual QA snapshots.
- `archive/data/`: committed dataset snapshots from the finished experiments.
- `archive/experiments/`: committed result tables, archived logs, smoke runs, calibration outputs, and diagnostic figures.
- `src/`: HSBO implementation, environments, baselines, and evaluation utilities.
- `scripts/`: grouped entry points for data generation, optional experiments,
  summaries, figures, and maintenance.
- `docs/`: structure and reproducibility notes.
- `_local/`: ignored local generated files moved out of the repository root.
- `outputs/` and `runs/`: ignored locations for any future exploratory reruns.

The repository is now organized for paper writing first. The completed
experiment artifacts are preserved under `archive/` instead of being spread
across the top level.

## Paper workflow

The active manuscript is:

```bash
paper/main.tex
```

It references figures under `paper/figures/`, so compile from the `paper/`
directory:

```bash
cd paper
latexmk -pdf main.tex
```

Selected polished WCL figures can be regenerated from archived summaries with:

```bash
python scripts/figures/plot_fig2_fig3_polished.py
python scripts/figures/plot_wireless_figures.py \
  --logs-root archive/experiments/logs \
  --results-root archive/experiments/results \
  --figures-root paper/figures
```

See `docs/PROJECT_STRUCTURE.md` for the full inventory and
`docs/REPRODUCIBILITY.md` for archived experiment commands. See
`scripts/README.md` and `src/README.md` for the code entry-point map.

## Optional reruns

Generate datasets:

```bash
python scripts/data/generate_datasets.py
```

Run a smoke test:

```bash
python scripts/experiments/run_experiment.py --scale small --methods random greedy_distance greedy_reward_density ga pso bo hsbo --budget 80 --seeds 0 1 --out outputs/v2_smoke
python scripts/summaries/summarize_results.py --results outputs/v2_smoke --out outputs/v2_smoke_summary.csv
python scripts/figures/plot_convergence.py --results outputs/v2_smoke --out outputs/v2_smoke_convergence.png
```

Run the 40-dimensional medium setting:

```bash
python scripts/experiments/run_experiment.py --scale medium --methods random greedy_distance greedy_reward_density ga pso bo hsbo --budget 150 --seeds 0 1 2 3 4 --out outputs/medium_main
python scripts/summaries/summarize_results.py --results outputs/medium_main --out outputs/medium_summary.csv
```

Run ablations:

```bash
python scripts/experiments/run_ablation.py --scale medium --budget 150 --seeds 0 1 2 3 4 --out outputs/medium_ablation
python scripts/summaries/summarize_results.py --results outputs/medium_ablation --out outputs/medium_ablation_summary.csv
```

## Wireless WCL experiment pipeline

The WCL-oriented experiment line is implemented separately from the original
sparse scheduling prototype. It models event-triggered emergency IoT packet
collection with hard validity windows and wireless rate/upload feasibility.

Generate wireless instances:

```bash
python scripts/data/generate_wireless_datasets.py --seeds 0 1 2 3 4 5 6 7 8 9
```

Run the 40-D calibration pilot:

```bash
python scripts/experiments/run_wireless_experiments.py --experiment pilot --samples 2000 --window-width 30 --rate-threshold-mbps 1
```

In the current evaluator, `W=30s, Rmin=1Mbps` is too easy for the 40-D random
pilot, while `Rmin=15Mbps` puts the random positive-utility ratio near the
target 0%--1% band. Use the calibrated threshold for the main run unless the
wireless parameters are changed:

```bash
python scripts/experiments/run_wireless_experiments.py --experiment sparsity --samples 2000 --rate-threshold-mbps 15
python scripts/experiments/run_wireless_experiments.py --experiment main --rate-threshold-mbps 15
python scripts/experiments/run_wireless_experiments.py --experiment ablation --rate-threshold-mbps 15
python scripts/experiments/run_wireless_experiments.py --experiment window --rate-threshold-mbps 15
python scripts/experiments/run_wireless_experiments.py --experiment rate --rate-values 10 15 20
python scripts/summaries/summarize_wireless_results.py --which all
python scripts/figures/plot_wireless_figures.py
```

By default, new wireless reruns write logs and summaries under `outputs/`.
Archived paper results are under `archive/experiments/`.

## Decision vector

For `N` UAVs and `K` maximum mission actions per UAV:

- Upper-level variables: `[speed_i, heading_i]_{i=1}^N`, dimension `2N`.
- Lower-level variables: `[task_key_{i,k}, scheduled_start_time_{i,k}]`, dimension `2NK`.

Total dimension: `2N + 2NK = 2N(K+1)`.

The default medium instance uses `N=5`, `K=3`, so the dimension is `40`.

## Recommended paper wording

Use cautious wording:

> We construct a reproducible sparse-reward cooperative scheduling extension inspired by public multi-UAV task-assignment and team-orienteering benchmarks. The extension introduces continuous scheduled-time variables, narrow time windows, and cooperative synchronization rewards.

Do **not** claim that these generated instances are the official public benchmark data. They are a reproducible extension. A later version can add an official benchmark loader.

## Current status

This is still a research prototype, not final submission code. Before paper submission, add stronger baselines such as ACO, CMA-ES, TuRBO, and optionally SAASBO.
