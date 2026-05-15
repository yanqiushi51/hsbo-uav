# Script Entry Points

Scripts are grouped by task so the repository stays readable during the
paper-writing stage.

## Data

- `data/generate_datasets.py`: regenerate sparse scheduling benchmark instances
  under `archive/data/datasets/`.
- `data/generate_wireless_datasets.py`: regenerate wireless WCL instances under
  `archive/data/datasets_wireless/`.

## Experiments

- `experiments/run_experiment.py`: run the original sparse scheduling benchmark.
- `experiments/run_ablation.py`: run sparse scheduling HSBO ablations.
- `experiments/run_enhanced_baselines.py`: run enhanced baselines such as CMA-ES
  and DE variants.
- `experiments/run_wireless_experiments.py`: run the wireless WCL experiments.

New experiment outputs default to ignored `outputs/` paths. Archived completed
results remain under `archive/experiments/`.

## Summaries

- `summaries/summarize_results.py`: summarize sparse scheduling CSV logs.
- `summaries/summarize_wireless_results.py`: summarize wireless WCL logs.

## Figures

- `figures/plot_fig2_fig3_polished.py`: regenerate the polished WCL paper
  figures from archived summaries.
- `figures/plot_wireless_figures.py`: generate wireless WCL diagnostic figures.
- `figures/plot_fig2_profile_convergence.py`: export best-so-far sequences and
  draw the Fig. 2 style profile.
- `figures/plot_enhanced_results.py`: summarize enhanced baseline comparisons.
- `figures/plot_convergence.py`, `figures/plot_paper_figures.py`, and
  `figures/redraw_paper_convergence.py`: legacy convergence plotting helpers.

## Maintenance

- `maintenance/clean_generated.py`: preview or remove ignored local generated
  files without touching tracked archived evidence.
