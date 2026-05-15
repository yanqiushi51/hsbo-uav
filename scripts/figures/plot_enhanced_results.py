#!/usr/bin/env python3
"""
Summarize enhanced baseline results alongside existing methods.

Produces a comparison CSV and a bar chart.

Usage:
  python scripts/figures/plot_enhanced_results.py \
    --scale-dirs 18-D=outputs/enhanced_small 40-D=outputs/enhanced_medium 80-D=outputs/enhanced_large \
    --out-prefix outputs/enhanced_summary
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Re-use the common figure utilities.
FIG_COMMON = ROOT / "paper" / "figure_scripts" / "hsbo_paper_figure_scripts" / "hsbo_figure_scripts"
if str(FIG_COMMON) not in sys.path:
    sys.path.insert(0, str(FIG_COMMON))

from hsbo_fig_common import (
    load_records, METHOD_ORDER, METHOD_LABEL, METHOD_COLOR,
)

# Augment labels and colours for enhanced methods.
ENHANCED_LABEL = {
    "cmaes": "CMA-ES",
    "cmaes_repair": "CMA-ES-r",
    "cmaes_seeded": "CMA-ES-p",
    "cmaes_repair_seeded": "CMA-ES-rp",
    "de": "DE",
    "de_repair": "DE-r",
    "de_seeded": "DE-p",
    "de_repair_seeded": "DE-rp",
}

ENHANCED_COLOR = {
    "cmaes": "#D8B365",
    "cmaes_repair": "#B2182B",
    "cmaes_seeded": "#E6C87A",
    "cmaes_repair_seeded": "#D4444F",
    "de": "#5EAFC0",
    "de_repair": "#2166AC",
    "de_seeded": "#8ECDD5",
    "de_repair_seeded": "#4B8DCB",
}


def summarise_dir(directory: str | Path, scale: str, budget: int | None = None) -> pd.DataFrame:
    df = load_records(Path(directory), scale, budget=budget)
    rows = []
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        n = len(sub)
        rows.append({
            "scale": scale,
            "method": method,
            "n_seeds": n,
            "final_best_mean": sub["final_best"].mean(),
            "final_best_std": sub["final_best"].std(ddof=1) if n > 1 else 0.0,
            "success_rate": 100.0 * (sub["final_best"] > 0).mean(),
            "feasible_rate": sub["feasible_rate"].mean() * 100.0,
        })
    return pd.DataFrame(rows)


def label_method(method: str) -> str:
    return ENHANCED_LABEL.get(method) or METHOD_LABEL.get(method, method)


def colour_method(method: str) -> str:
    return ENHANCED_COLOR.get(method) or METHOD_COLOR.get(method, "#888888")


def plot(summary: pd.DataFrame, out_prefix: str):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
    })

    scales = ["18-D", "40-D", "80-D"]
    methods = summary["method"].unique().tolist()
    # Sort: HSBO first, then enhanced, then original baselines.
    priority = {m: i for i, m in enumerate(METHOD_ORDER + ["cmaes", "cmaes_repair", "de", "de_repair"])}
    methods = sorted(methods, key=lambda m: priority.get(m, 99))

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))

    # ---- (a) final best reward ----
    ax = axes[0]
    n_scales = len(scales)
    n_methods = len(methods)
    width = 0.75 / n_methods
    x = np.arange(n_scales)

    for idx, method in enumerate(methods):
        vals = []
        for scale in scales:
            row = summary[(summary["scale"] == scale) & (summary["method"] == method)]
            vals.append(row["final_best_mean"].values[0] if len(row) else 0.0)
        offset = (idx - (n_methods - 1) / 2) * width
        color = colour_method(method)
        ax.bar(x + offset, vals, width=width, color=color, edgecolor="white", linewidth=0.3,
               label=label_method(method))

    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_ylabel("Final best reward")
    ax.set_title("(a) Final best reward", fontsize=10, fontweight="normal")
    ax.grid(axis="y", linewidth=0.3, alpha=0.3, color="#BBBBBB")
    ax.legend(frameon=False, ncol=2, fontsize=6.5, loc="upper left")

    # ---- (b) success rate ----
    ax = axes[1]
    for idx, method in enumerate(methods):
        vals = []
        for scale in scales:
            row = summary[(summary["scale"] == scale) & (summary["method"] == method)]
            vals.append(row["success_rate"].values[0] if len(row) else np.nan)
        offset = (idx - (n_methods - 1) / 2) * width
        color = colour_method(method)
        ax.bar(x + offset, vals, width=width, color=color, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("(b) Feasible discovery success rate", fontsize=10, fontweight="normal")
    ax.grid(axis="y", linewidth=0.3, alpha=0.3, color="#BBBBBB")

    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{out_prefix}_comparison.png", dpi=600, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    # Save CSV.
    csv_path = f"{out_prefix}_comparison.csv"
    summary.to_csv(csv_path, index=False, float_format="%.3f")
    print(f"Saved {csv_path}")
    print(f"Saved {out_prefix}_comparison.png")


def _parse_scale_dir(items):
    """Parse SCALE=DIR pairs like '18-D=outputs/enhanced_small'."""
    result = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected SCALE=DIR, got '{item}'")
        scale, d = item.split("=", 1)
        result.append((scale, Path(d)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-dirs", nargs="+", required=True,
                        help="SCALE=DIR pairs, e.g. '18-D=outputs/enhanced_small'")
    parser.add_argument("--out-prefix", default="outputs/enhanced_summary")
    parser.add_argument("--budget", type=int, default=None)
    args = parser.parse_args()

    frames = []
    for scale, directory in _parse_scale_dir(args.scale_dirs):
        frames.append(summarise_dir(directory, scale, budget=args.budget))

    summary = pd.concat(frames, ignore_index=True)
    plot(summary, args.out_prefix)


if __name__ == "__main__":
    main()
