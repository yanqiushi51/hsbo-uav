#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4: budget sensitivity on the 40-D instance (JSEE-ready).

Uses trajectory-truncation from a single b=800 run, so the best-so-far
curve is monotone non-decreasing — the correct anytime behaviour.

Prerequisite:
  python scripts/experiments/run_experiment.py \
    --scale medium \
    --methods random greedy_distance greedy_reward_density ga pso bo hsbo \
    --budget 800 --seeds 0 1 2 3 4 \
    --out outputs/budget_medium_b800

Usage:
  python plot_budget_sensitivity_40d.py \
    --data-dir archive/experiments/results/budget_medium_b800 \
    --budgets 100 200 300 500 800 \
    --out paper/figures/fig4_budget_sensitivity_40d.png
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hsbo_fig_common import load_records, METHOD_ORDER, label, METHOD_COLOR

# Only the methods present in the budget-sensitivity experiments.
BUDGET_METHODS = ["hsbo", "pso", "bo", "random", "ga", "greedy_distance", "greedy_reward_density"]


def build_table(data_dir: Path, budgets: List[int]) -> dict:
    """Load b=800 data and truncate at each budget to produce monotone curves."""
    df_full = load_records(data_dir, "40-D", budget=None)   # full 800-row trajectories
    rows = []
    for budget in budgets:
        # Truncate each seed's trajectory to the first `budget` evaluations.
        df_clip = load_records(data_dir, "40-D", budget=budget)
        for method in df_clip["method"].unique():
            sub = df_clip[df_clip["method"] == method]
            n = len(sub)
            rows.append({
                "budget": budget,
                "method": method,
                "mean": sub["final_best"].mean(),
                "std": sub["final_best"].std(ddof=1) if n > 1 else 0.0,
                "success_rate": 100.0 * (sub["final_best"] > 0).mean(),
                "n_seeds": n,
            })
    import pandas as pd
    return pd.DataFrame(rows)


def plot(table, budgets: List[int], out: Path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3))

    # ---- (a) Final best reward ----
    ax = axes[0]
    for method in BUDGET_METHODS:
        sub = table[table["method"] == method].sort_values("budget")
        x = sub["budget"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        s = sub["std"].to_numpy(dtype=float)

        color = METHOD_COLOR.get(method, "#888888")
        lw = 1.8 if method == "hsbo" else 1.0
        ms = 4.5 if method == "hsbo" else 3.0
        zorder = 5 if method == "hsbo" else 1
        alpha = 1.0 if method == "hsbo" else 0.65

        ax.plot(x, y, marker="o", linewidth=lw, markersize=ms,
                color=color, label=label(method), zorder=zorder, alpha=alpha)
        ax.fill_between(x, y - s, y + s, alpha=0.10, color=color, linewidth=0)

    # Mark PSO outlier at b=800.
    pso_row = table[(table["method"] == "pso") & (table["budget"] == 800)]
    if len(pso_row) and pso_row["mean"].values[0] > 0:
        sr = pso_row["success_rate"].values[0]
        mv = pso_row["mean"].values[0]
        ax.annotate(
            f'PSO: {sr:.0f}% ({mv:.1f})',
            xy=(800, mv),
            xytext=(620, 12),
            fontsize=6.5, color=METHOD_COLOR["pso"],
            arrowprops=dict(arrowstyle="->", color=METHOD_COLOR["pso"], lw=0.7),
            ha="center",
        )

    ax.axhline(y=0, linewidth=0.4, color="#CCCCCC", zorder=0)
    ax.set_xlabel("Evaluation budget")
    ax.set_ylabel("Final best reward")
    ax.set_title("(a) Final best reward", fontsize=10, fontweight="normal")
    ax.grid(True, linewidth=0.3, alpha=0.3, color="#BBBBBB")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.50, 1.18),
        ncol=4, frameon=False, fontsize=7,
        handlelength=1.5, handletextpad=0.4, columnspacing=0.8,
    )

    # ---- (b) Success rate ----
    ax = axes[1]
    for method in BUDGET_METHODS:
        sub = table[table["method"] == method].sort_values("budget")
        x = sub["budget"].to_numpy(dtype=float)
        y = sub["success_rate"].to_numpy(dtype=float)

        color = METHOD_COLOR.get(method, "#888888")
        lw = 1.8 if method == "hsbo" else 1.0
        ms = 4.5 if method == "hsbo" else 3.0
        zorder = 5 if method == "hsbo" else 1
        alpha = 1.0 if method == "hsbo" else 0.65

        ax.plot(x, y, marker="o", linewidth=lw, markersize=ms,
                color=color, label=label(method), zorder=zorder, alpha=alpha)

    ax.axhline(y=0, linewidth=0.4, color="#CCCCCC", zorder=0)
    ax.set_xlabel("Evaluation budget")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(-3, 110)
    ax.set_title("(b) Feasible discovery success rate", fontsize=10, fontweight="normal")
    ax.grid(True, linewidth=0.3, alpha=0.3, color="#BBBBBB")

    fig.tight_layout(w_pad=1.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Directory with b=800 per-seed CSV files.")
    parser.add_argument("--budgets", nargs="+", type=int,
                        default=[100, 200, 300, 500, 800])
    parser.add_argument("--out", default="paper/figures/fig4_budget_sensitivity_40d.png")
    args = parser.parse_args()

    table = build_table(Path(args.data_dir), args.budgets)
    plot(table, args.budgets, Path(args.out))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
