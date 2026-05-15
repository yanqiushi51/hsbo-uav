#!/usr/bin/env python3
"""
Redraw paper-quality convergence figures for medium (40-D) and large (80-D).

Requirements:
  - HSBO: solid line + standard-error band.
  - All baselines: merged into one gray dashed zero line when all are zero.
  - X-axis from 1. Labels: "Number of objective evaluations".
  - Y-axis: "Best-so-far reward".
  - Output: 600 dpi.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
    "figure.dpi": 150,
})


BASELINE_METHODS = ["bo", "ga", "greedy_distance", "greedy_reward_density",
                    "pso", "random"]


def _load_best_curves(result_dir: Path, method: str, budget: int) -> np.ndarray | None:
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


def _baselines_all_zero(result_dir: Path, budget: int) -> bool:
    for m in BASELINE_METHODS:
        c = _load_best_curves(result_dir, m, budget)
        if c is not None and c.max() > 0.05:
            return False
    return True


def draw_one(result_dir: str, budget: int, out: str, title_suffix: str):
    result_dir = Path(result_dir)
    x = np.arange(1, budget + 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # ---------- HSBO ----------
    hsbo = _load_best_curves(result_dir, "hsbo", budget)
    if hsbo is not None and hsbo.shape[0] > 0:
        mu = hsbo.mean(axis=0)
        se = hsbo.std(axis=0) / np.sqrt(hsbo.shape[0])
        ax.plot(x, mu, linewidth=2.4, color="#1f77b4", label="HSBO")
        ax.fill_between(x, mu - se, mu + se, alpha=0.16, color="#1f77b4")

    # ---------- Baselines ----------
    if _baselines_all_zero(result_dir, budget):
        ax.plot(x, np.zeros_like(x), linestyle="--", linewidth=1.6,
                color="gray", label="All baselines")
    else:
        for m in BASELINE_METHODS:
            c = _load_best_curves(result_dir, m, budget)
            if c is None:
                continue
            ax.plot(x, c.mean(axis=0), linewidth=1.2, alpha=0.7,
                    label=m.replace("_", "-"))

    # ---------- Labels ----------
    ax.set_xlabel("Number of objective evaluations")
    ax.set_ylabel("Best-so-far reward")
    ax.set_xlim(1, budget)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"Convergence on {title_suffix} instance",
                 fontsize=13, fontweight="normal", pad=8)

    fig.tight_layout()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--medium-dir", type=str, required=True)
    parser.add_argument("--large-dir", type=str, required=True)
    parser.add_argument("--medium-budget", type=int, required=True)
    parser.add_argument("--large-budget", type=int, required=True)
    parser.add_argument("--out-medium", type=str, required=True)
    parser.add_argument("--out-large", type=str, required=True)
    args = parser.parse_args()

    draw_one(args.medium_dir, args.medium_budget, args.out_medium, "40-D")
    draw_one(args.large_dir, args.large_budget, args.out_large, "80-D")


if __name__ == "__main__":
    main()
