#!/usr/bin/env python3
"""
Paper-quality convergence plot generator.

Key fixes over the simple version:
  1. X-axis starts from evaluation 1 (not 0).
  2. All-zero baselines merged into one gray dashed "All baselines" line.
  3. HSBO curve shown as mean ± standard error band.
  4. 600 dpi output.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Style defaults suitable for JSEE
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
    "figure.dpi": 150,
})


def load_best_curves(result_dir: Path, method: str, budget: int) -> np.ndarray | None:
    """Return (n_seeds, budget) array of best-so-far rewards for one method."""
    files = sorted(result_dir.glob(f"*_{method}_seed*.csv"))
    if not files:
        return None
    curves = []
    for f in files:
        df = pd.read_csv(f)
        col = "best_reward" if "best_reward" in df.columns else "reward"
        reward = df[col].to_numpy(dtype=float)[:budget]
        if len(reward) < budget:
            reward = np.pad(reward, (0, budget - len(reward)),
                            constant_values=reward[-1] if len(reward) else 0.0)
        curves.append(reward)
    return np.array(curves)


def plot_paper_convergence(result_dir: str, out: str, budget: int,
                           title_suffix: str = ""):
    result_dir = Path(result_dir)
    methods = ["bo", "ga", "greedy_distance", "greedy_reward_density",
               "pso", "random", "hsbo"]

    x = np.arange(1, budget + 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # --- HSBO with standard error band ---
    hsbo = load_best_curves(result_dir, "hsbo", budget)
    if hsbo is not None and hsbo.shape[0] > 0:
        mu = hsbo.mean(axis=0)
        se = hsbo.std(axis=0) / np.sqrt(hsbo.shape[0])
        ax.plot(x, mu, linewidth=2.4, color="#1f77b4", label="HSBO")
        ax.fill_between(x, mu - se, mu + se, alpha=0.16, color="#1f77b4")

    # --- Baselines: merge all-zero or plot individually ---
    baseline_methods = [m for m in methods if m != "hsbo"]
    all_zero = True
    baseline_best = []
    for m in baseline_methods:
        c = load_best_curves(result_dir, m, budget)
        if c is not None:
            m_max = c.max()
            if m_max > 0.05:
                all_zero = False
            baseline_best.append((m, c))

    if all_zero:
        ax.plot(x, np.zeros_like(x), linestyle="--", linewidth=1.6,
                color="gray", label="All baselines")
    else:
        for m, c in baseline_best:
            mu = c.mean(axis=0)
            ax.plot(x, mu, linewidth=1.2, alpha=0.7, label=m.replace("_", "-"))

    # --- Axis labels and grid ---
    ax.set_xlabel("Number of objective evaluations")
    ax.set_ylabel("Best-so-far reward")
    ax.set_xlim(1, budget)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, loc="lower right")

    title = f"Convergence on {title_suffix} instance"
    ax.set_title(title, fontsize=13, fontweight="normal", pad=8)

    fig.tight_layout()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()
    plot_paper_convergence(args.results, args.out, args.budget, args.title)


if __name__ == "__main__":
    main()
