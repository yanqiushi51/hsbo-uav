# Reproducibility Guide

The project is currently in the paper-writing stage. Archived experiment
artifacts live under `archive/`, while new reruns should write to ignored
`outputs/` directories.

## Environment

```bash
conda create -n hsbo-uav python=3.10 -y
conda activate hsbo-uav
pip install -r requirements.txt
pip install -r requirements_extra.txt
```

`requirements_extra.txt` is only needed for optional enhanced baselines such as
CMA-ES.

## Manuscript

The active manuscript is `paper/main.tex`. Compile from the paper directory:

```bash
cd paper
latexmk -pdf main.tex
```

The manuscript references:

- `paper/figures/fig1.pdf`
- `paper/figures/fig2.pdf`
- `paper/figures/fig3_sidebyside.pdf`

## Archived Evidence

- Dataset snapshots: `archive/data/`
- Result snapshots: `archive/experiments/results*`
- Archived logs: `archive/experiments/logs*`
- Paper figure exports: `paper/data/export/`

## Optional Sparse Scheduling Rerun

```bash
python scripts/data/generate_datasets.py

python scripts/experiments/run_experiment.py \
  --scale small \
  --methods random greedy_distance greedy_reward_density ga pso bo hsbo \
  --budget 80 \
  --seeds 0 1 \
  --out outputs/sparse_smoke

python scripts/summaries/summarize_results.py \
  --results outputs/sparse_smoke \
  --out outputs/sparse_smoke_summary.csv
```

## Optional Wireless WCL Rerun

```bash
python scripts/data/generate_wireless_datasets.py --seeds 0 1 2 3 4 5 6 7 8 9

python scripts/experiments/run_wireless_experiments.py \
  --experiment main \
  --rate-threshold-mbps 15

python scripts/summaries/summarize_wireless_results.py --which all

python scripts/figures/plot_wireless_figures.py
```

By default, these commands use `archive/data/datasets_wireless` for dataset
snapshots and write new logs/results/figures to ignored `outputs/` or
`paper/figures/` paths.

## Regenerating Polished Paper Figures

```bash
python scripts/figures/plot_fig2_fig3_polished.py

python scripts/figures/plot_wireless_figures.py \
  --logs-root archive/experiments/logs \
  --results-root archive/experiments/results \
  --figures-root paper/figures
```

Additional legacy figure helpers live under `paper/figure_scripts/`.

## Cleaning Local Generated Files

```bash
python scripts/maintenance/clean_generated.py --dry-run
python scripts/maintenance/clean_generated.py --apply
```

The cleanup script only removes Git-ignored paths. It does not remove tracked
archived evidence.
