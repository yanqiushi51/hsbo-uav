#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2: multiscale final-best reward distribution (JSEE-ready).

python plot_multiscale_reward_distribution.py \
  --small-dir results/v2_small_10seeds \
  --medium-dir results/v2_medium_b300 \
  --large-dir results/v2_large_b300 \
  --out figures/fig2_reward_distribution.png
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

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=False)

    for ax, scale in zip(axes, scales):
        sub = df[df["scale"] == scale]
        positions = np.arange(1, len(methods) + 1)

        for i, method in enumerate(methods):
            vals = sub[sub["method"] == method]["final_best"].to_numpy()
            color = METHOD_COLOR.get(method, "#888888")
            pos = i + 1

            if len(vals) == 0:
                continue

            bp = ax.boxplot(
                [vals],
                positions=[pos],
                widths=0.55,
                showfliers=False,
                patch_artist=True,
                medianprops={"linewidth": 1.0, "color": color},
                whiskerprops={"linewidth": 0.7, "color": color},
                capprops={"linewidth": 0.7, "color": color},
                boxprops={"linewidth": 0.7, "edgecolor": color, "facecolor": "none"},
            )

            rng = np.random.default_rng(2026)
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                np.full_like(vals, pos, dtype=float) + jitter,
                vals,
                s=10, alpha=0.65,
                facecolor=color, edgecolor="none", linewidth=0,
            )

        ax.axhline(y=0, linewidth=0.4, color="#CCCCCC", zorder=0)

        ax.set_title(scale, pad=5, fontsize=10, fontweight="normal")
        ax.set_xticks(positions)
        ax.set_xticklabels([label(m) for m in methods], rotation=0, ha="center")
        ax.set_xlim(0.2, len(methods) + 0.8)
        ax.set_ylabel("Final best reward" if scale == "18-D" else "")
        ax.grid(axis="y", linewidth=0.3, alpha=0.3, color="#BBBBBB")

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(min(-2, ymin), ymax * 1.06 if ymax > 0 else 1)

    fig.tight_layout(w_pad=1.2)
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
    parser.add_argument("--out", default="fig2_reward_distribution.png")
    args = parser.parse_args()

    df = load_multiscale(
        args.small_dir, args.medium_dir, args.large_dir,
        args.small_budget, args.medium_budget, args.large_budget,
    )
    plot(df, Path(args.out))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
