# HSBO figure scripts

These scripts generate more informative paper figures for the HSBO paper.

## 1. Multiscale final-best distribution

```bash
python plot_multiscale_reward_distribution.py \
  --small-dir archive/experiments/results/v2_small_10seeds \
  --medium-dir archive/experiments/results/v2_medium_b300 \
  --large-dir archive/experiments/results/v2_large_b300 \
  --out paper/figures/fig2_reward_distribution.png
```

Caption suggestion:

Fig. 2 Distribution of final best rewards on the 18-D, 40-D, and 80-D instances. Each point corresponds to one independent seed. The box plots show the median and interquartile range. HSBO consistently obtains higher final best rewards, whereas baseline methods collapse to zero in the 40-D and 80-D settings.

## 2. Success rate and first feasible evaluation

```bash
python plot_success_first_feasible.py \
  --small-dir archive/experiments/results/v2_small_10seeds \
  --medium-dir archive/experiments/results/v2_medium_b300 \
  --large-dir archive/experiments/results/v2_large_b300 \
  --out paper/figures/fig3_success_first_feasible.png
```

Caption suggestion:

Fig. 3 Feasible-schedule discovery analysis. (a) Success rate, defined as the percentage of seeds that discover at least one positive-reward schedule. (b) First feasible evaluation index of HSBO on different scales. The results show that HSBO is not only better in final reward but also more effective at entering sparse feasible regions.

## 3. Budget sensitivity on the 40-D instance

First run additional experiments:

```bash
for b in 100 200 300 500 800; do
  python scripts/experiments/run_experiment.py \
    --scale medium \
    --methods random ga pso bo hsbo \
    --budget $b \
    --seeds 0 1 2 3 4 \
    --out outputs/budget_medium_b${b}
done
```

Then plot:

```bash
python plot_budget_sensitivity_40d.py \
  --budget-dirs \
    100=archive/experiments/results/budget_medium_b100 \
    200=archive/experiments/results/budget_medium_b200 \
    300=archive/experiments/results/budget_medium_b300 \
    500=archive/experiments/results/budget_medium_b500 \
    800=archive/experiments/results/budget_medium_b800 \
  --out paper/figures/fig4_budget_sensitivity_40d.png
```

Caption suggestion:

Fig. 4 Budget sensitivity on the 40-D instance. Curves show the final best reward under different objective-evaluation budgets. HSBO benefits from larger budgets and maintains a clear advantage, while full-space and population-based baselines remain unable to reliably discover positive-reward schedules.

## Notes

- All plots use per-seed CSV logs rather than summary-only files.
- The scripts first compute each seed's best-so-far trajectory or final best reward, then aggregate across seeds.
- Do not point these scripts to summary-only directories.
