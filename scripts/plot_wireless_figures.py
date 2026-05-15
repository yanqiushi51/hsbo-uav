from __future__ import annotations

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


METHOD_ORDER = [
    "ft_hsbo",
    "full_space_bo",
    "de_rp",
    "edf_rate_greedy",
    "max_rate_greedy",
    "pso",
    "ga",
    "rs",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Wrote {path}")


def _best_by_seed(logs_root: Path, scale: str) -> pd.DataFrame:
    frames = []
    for path in (logs_root / "main" / scale).glob("*/seed_*.csv"):
        df = pd.read_csv(path)
        if df.empty:
            continue
        frames.append(
            {
                "method": str(df["method"].iloc[0]),
                "seed": int(df["seed"].iloc[0]),
                "best": float(df["best_so_far"].iloc[-1]),
            }
        )
    return pd.DataFrame(frames)


def plot_main_boxplot(logs_root: Path, figures_root: Path, scale: str, filename: str, title: str):
    df = _best_by_seed(logs_root, scale)
    if df.empty:
        return
    methods = [m for m in METHOD_ORDER if m in set(df["method"])] + [m for m in sorted(df["method"].unique()) if m not in METHOD_ORDER]
    data = [df.loc[df["method"] == m, "best"].values for m in methods]
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(methods)), 4.4))
    ax.boxplot(data, tick_labels=methods, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel("Best utility")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, figures_root / filename)


def plot_summary_bar(summary: pd.DataFrame, value: str, filename: Path, title: str, ylabel: str):
    if summary.empty or value not in summary.columns:
        return
    df = summary.copy()
    df["label"] = df["scale"].astype(str) + " / " + df["method"].astype(str)
    fig, ax = plt.subplots(figsize=(max(7, 0.42 * len(df)), 4.3))
    ax.bar(np.arange(len(df)), df[value].astype(float))
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, filename)


def plot_method_lines(summary: pd.DataFrame, xcol: str, ycol: str, filename: Path, title: str, xlabel: str):
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for method, g in summary.groupby("method"):
        g = g.sort_values(xcol)
        ax.plot(g[xcol], g[ycol], marker="o", label=method)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean utility")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, filename)


def plot_channel(summary: pd.DataFrame, filename: Path):
    if summary.empty:
        return
    pivot = summary.pivot_table(index="channel_model", columns="method", values="mean_utility", aggfunc="mean")
    methods = [m for m in METHOD_ORDER if m in pivot.columns] + [m for m in pivot.columns if m not in METHOD_ORDER]
    pivot = pivot[methods]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Channel robustness")
    ax.set_ylabel("Mean utility")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--figures-root", default="figures")
    args = parser.parse_args()

    logs_root = ROOT / args.logs_root
    results_root = ROOT / args.results_root
    figures_root = ROOT / args.figures_root

    plot_main_boxplot(logs_root, figures_root, "medium", "main_boxplot_40D.png", "Main comparison: 40-D")
    plot_main_boxplot(logs_root, figures_root, "large", "main_boxplot_80D.png", "Main comparison: 80-D")

    main_summary = _read_csv(results_root / "summary_main.csv")
    plot_summary_bar(main_summary, "success_rate", figures_root / "discovery_success.png", "Discovery success", "Success rate")
    plot_summary_bar(main_summary, "median_first_feasible", figures_root / "first_feasible_index.png", "First feasible index", "Median evaluation index")

    ablation = _read_csv(results_root / "summary_ablation.csv")
    plot_summary_bar(ablation, "mean_utility", figures_root / "ablation_bar.png", "Ablation study", "Mean utility")

    window = _read_csv(results_root / "summary_sensitivity_window.csv")
    plot_method_lines(window, "window_width", "mean_utility", figures_root / "window_sensitivity.png", "Window-width sensitivity", "Window width (s)")

    rate = _read_csv(results_root / "summary_sensitivity_rate.csv")
    plot_method_lines(rate, "rate_threshold_mbps", "mean_utility", figures_root / "rate_sensitivity.png", "Rate-threshold sensitivity", "Rate threshold (Mbps)")

    channel = _read_csv(results_root / "summary_channel.csv")
    plot_channel(channel, figures_root / "channel_robustness.png")

    budget = _read_csv(results_root / "summary_budget.csv")
    plot_method_lines(budget, "budget", "mean_utility", figures_root / "budget_sensitivity.png", "Budget sensitivity", "Budget")


if __name__ == "__main__":
    main()
