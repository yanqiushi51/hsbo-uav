#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3: feasible discovery analysis (JSEE-ready).

python plot_success_first_feasible.py \
  --small-dir results/v2_small_10seeds \
  --medium-dir results/v2_medium_b300 \
  --large-dir results/v2_large_b300 \
  --out figures/fig3_success_first_feasible.png
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hsbo_fig_common import load_multiscale, ordered_methods, label, METHOD_COLOR


def plot(df, out: Path):
    scales = ["18-D", "40-D", "80-D"]
    methods = ordered_methods(df)
    hsbo_color = METHOD_COLOR["hsbo"]
    baseline_color = "#AAAAAA"

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # ---- (a) Success rate ----
    ax = axes[0]
    n_scales = len(scales)
    n_methods = len(methods)
    width = 0.12
    x = np.arange(n_scales)

    for idx, method in enumerate(methods):
        vals = []
        for scale in scales:
            sub = df[(df["scale"] == scale) & (df["method"] == method)]
            vals.append(100.0 * (sub["final_best"] > 0).mean() if len(sub) else np.nan)

        color = hsbo_color if method == "hsbo" else baseline_color
        offset = (idx - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width, color=color, edgecolor="white", linewidth=0.3)

        if method == "hsbo":
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                            f"{v:.0f}%", ha="center", va="bottom", fontsize=6.5,
                            color=hsbo_color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("(a) Feasible discovery success", pad=6, fontsize=10, fontweight="normal")
    ax.grid(axis="y", linewidth=0.3, alpha=0.3, color="#BBBBBB")

    h_leg = [plt.Rectangle((0, 0), 1, 1, fc=hsbo_color)]
    b_leg = [plt.Rectangle((0, 0), 1, 1, fc=baseline_color)]
    leg_labels = ["HSBO", "Baselines"]
    ax.legend(h_leg + b_leg, leg_labels, frameon=False, ncol=2,
              loc="upper left", fontsize=7.5, handlelength=1.2, handleheight=1.0)

    # ---- (b) HSBO first feasible ----
    ax = axes[1]
    data = []
    ticklabels = []
    stats = []
    for scale in scales:
        sub = df[(df["scale"] == scale) & (df["method"] == "hsbo")]
        vals = sub["first_feasible"].dropna().astype(float).to_numpy()
        data.append(vals)
        ticklabels.append(scale)
        stats.append((vals.mean(), vals.min(), vals.max()) if len(vals) > 0 else (np.nan, np.nan, np.nan))

    rng = np.random.default_rng(2026)
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(
            np.full_like(vals, i, dtype=float) + jitter,
            vals,
            s=28, alpha=0.75,
            facecolor=hsbo_color, edgecolor="white", linewidth=0.4,
            zorder=3,
        )

    for i, (mean, vmin, vmax) in enumerate(stats, start=1):
        if np.isnan(mean):
            continue
        ax.plot([i - 0.25, i + 0.25], [mean, mean],
                linewidth=2.2, color="#B2182B", solid_capstyle="round", zorder=4)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(ticklabels)
    ax.set_ylabel("First feasible evaluation")
    ax.set_title("(b) HSBO first feasible hit", pad=6, fontsize=10, fontweight="normal")
    ax.grid(axis="y", linewidth=0.3, alpha=0.3, color="#BBBBBB")

    fig.tight_layout(w_pad=1.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-dir", required=True)
    parser.add_argument("--medium-dir", required=True)
    parser.add_argument("--large-dir", required=True)
    parser.add_argument("--small-budget", type=int, default=None)
    parser.add_argument("--medium-budget", type=int, default=None)
    parser.add_argument("--large-budget", type=int, default=None)
    parser.add_argument("--out", default="fig3_success_first_feasible.png")
    args = parser.parse_args()

    df = load_multiscale(
        args.small_dir, args.medium_dir, args.large_dir,
        args.small_budget, args.medium_budget, args.large_budget,
    )
    plot(df, Path(args.out))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
