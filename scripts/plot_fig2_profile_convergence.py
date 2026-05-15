from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

METHOD_LABELS = {
    "ft_hsbo": "FT-HSBO",
    "max_rate_greedy": "Max-Rate",
    "repair_bo": "Repair BO",
    "de_rp": "DE-rp",
    "full_space_bo": "Full BO",
    "rs": "RS",
}

METHOD_ORDER = [
    "ft_hsbo",
    "max_rate_greedy",
    "repair_bo",
    "de_rp",
    "full_space_bo",
    "rs",
]

COLORS = {
    "ft_hsbo": "#1f77b4",
    "max_rate_greedy": "#ff7f0e",
    "repair_bo": "#2ca02c",
    "de_rp": "#d62728",
    "full_space_bo": "#9467bd",
    "rs": "#6f6f6f",
}

LINESTYLES = {
    "ft_hsbo": "-",
    "max_rate_greedy": "--",
    "repair_bo": "--",
    "de_rp": "-.",
    "full_space_bo": ":",
    "rs": (0, (5, 2)),
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "legend.fontsize": 5.8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def parse_seed(path: Path, df: pd.DataFrame) -> int:
    if "seed" in df.columns and len(df["seed"].dropna()) > 0:
        return int(df["seed"].dropna().iloc[0])
    match = re.search(r"seed_(\d+)", path.stem)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer seed from {path}")


def load_run(path: Path, method_id: str, scale: str, budget: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "eval_idx" in df.columns:
        df = df.sort_values("eval_idx")
        eval_idx = df["eval_idx"].astype(int).to_numpy()
    else:
        eval_idx = np.arange(1, len(df) + 1)

    if "best_so_far" in df.columns:
        best = df["best_so_far"].astype(float).to_numpy()
    elif "best_reward" in df.columns:
        best = df["best_reward"].astype(float).to_numpy()
    elif "utility" in df.columns:
        best = np.maximum.accumulate(df["utility"].astype(float).to_numpy())
    else:
        raise ValueError(f"No best-so-far or utility column in {path}")

    if "utility" in df.columns:
        utility = df["utility"].astype(float).to_numpy()
    elif "current_reward" in df.columns:
        utility = df["current_reward"].astype(float).to_numpy()
    else:
        utility = best.copy()

    run = pd.DataFrame({"evaluation": eval_idx, "utility": utility, "best_so_far": best})
    run = run.drop_duplicates("evaluation", keep="last").set_index("evaluation")
    run = run.reindex(np.arange(1, budget + 1))
    run["best_so_far"] = run["best_so_far"].ffill().fillna(0.0)
    run["utility"] = run["utility"].fillna(0.0)
    run = run.reset_index().rename(columns={"index": "evaluation"})

    seed = parse_seed(path, df)
    run.insert(0, "seed", seed)
    run.insert(0, "scale", scale)
    run.insert(0, "method", METHOD_LABELS.get(method_id, method_id))
    run.insert(0, "method_id", method_id)
    run["success_so_far"] = run["best_so_far"] > 0.0
    run["source_log"] = str(path.relative_to(ROOT)).replace("\\", "/")
    return run


def collect_sequences(logs_root: Path, scale: str, budget: int, methods: list[str]) -> pd.DataFrame:
    rows = []
    scale_root = logs_root / "main" / scale
    for method_id in methods:
        method_root = scale_root / method_id
        if not method_root.exists():
            print(f"Skipping missing method directory: {method_root}")
            continue
        for path in sorted(method_root.glob("seed_*.csv")):
            rows.append(load_run(path, method_id, scale, budget))
    if not rows:
        raise FileNotFoundError(f"No seed CSV files found under {scale_root}")
    seq = pd.concat(rows, ignore_index=True)
    seq["method_id"] = pd.Categorical(seq["method_id"], methods, ordered=True)
    seq = seq.sort_values(["method_id", "seed", "evaluation"]).reset_index(drop=True)
    return seq


def summarize_sequences(seq: pd.DataFrame) -> pd.DataFrame:
    grouped = seq.groupby(["method_id", "method", "scale", "evaluation"], observed=True)
    summary = grouped.agg(
        n_runs=("seed", "nunique"),
        success_rate=("success_so_far", "mean"),
        mean_best_so_far=("best_so_far", "mean"),
        std_best_so_far=("best_so_far", "std"),
        q25_best_so_far=("best_so_far", lambda s: float(np.quantile(s, 0.25))),
        q75_best_so_far=("best_so_far", lambda s: float(np.quantile(s, 0.75))),
    ).reset_index()
    summary["std_best_so_far"] = summary["std_best_so_far"].fillna(0.0)
    summary["sem_best_so_far"] = summary["std_best_so_far"] / np.sqrt(summary["n_runs"])
    return summary


def plot_fig2(summary: pd.DataFrame, out_stem: Path, methods: list[str]) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.55, 2.05), sharex=True)

    plot_order = [m for m in methods if m not in {"ft_hsbo", "max_rate_greedy"}]
    plot_order += [m for m in ["ft_hsbo", "max_rate_greedy"] if m in methods]
    max_eval = int(summary["evaluation"].max())

    for method_id in plot_order:
        sub = summary[summary["method_id"].astype(str) == method_id].sort_values("evaluation")
        if sub.empty:
            continue
        x = sub["evaluation"].to_numpy(dtype=float)
        label = METHOD_LABELS.get(method_id, method_id)
        color = COLORS.get(method_id, "#333333")
        linestyle = LINESTYLES.get(method_id, "-")
        linewidth = 1.35
        if method_id in {"full_space_bo", "rs"}:
            linewidth = 1.05
        alpha = 1.0 if method_id in {"ft_hsbo", "max_rate_greedy"} else 0.85
        zorder = 10 if method_id == "max_rate_greedy" else 8 if method_id == "ft_hsbo" else 3

        axes[0].plot(
            x,
            sub["success_rate"].to_numpy(dtype=float),
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder,
        )

        y = sub["mean_best_so_far"].to_numpy(dtype=float)
        se = sub["sem_best_so_far"].to_numpy(dtype=float)
        axes[1].plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder,
        )
        band_alpha = 0.16 if method_id in {"ft_hsbo", "max_rate_greedy"} else 0.08
        axes[1].fill_between(x, y - se, y + se, color=color, alpha=band_alpha, linewidth=0, zorder=max(1, zorder - 2))

    axes[0].set_title("(a) Discovery")
    axes[0].set_xlabel("Evaluations")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(-0.03, 1.06)
    axes[0].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    axes[1].set_title("(b) Best-so-far utility")
    axes[1].set_xlabel("Evaluations")
    axes[1].set_ylabel("Mean utility")
    ymax = summary["mean_best_so_far"].max()
    axes[1].set_ylim(bottom=0, top=max(1.0, ymax * 1.18))

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(1, max_eval)
        ticks = [1, 2, 4, 8, 16, 32, 100, 300]
        ticks = [t for t in ticks if t <= max_eval]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.grid(True, color="#d7dce2", linewidth=0.45, alpha=0.9)
        ax.set_axisbelow(True)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncols=3,
        frameon=False,
        handlelength=1.45,
        columnspacing=0.55,
        handletextpad=0.25,
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.74, bottom=0.22, wspace=0.38)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".png", ".svg"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 500
        out = out_stem.with_suffix(suffix)
        fig.savefig(out, **kwargs)
        print(f"Wrote {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export evaluation-wise best-so-far sequences and draw Fig. 2."
    )
    parser.add_argument("--logs-root", type=Path, default=ROOT / "logs")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--figures-root", type=Path, default=ROOT / "figures")
    parser.add_argument("--scale", default="medium", choices=["medium", "large"])
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--methods", nargs="*", default=METHOD_ORDER)
    parser.add_argument("--figure-stem", default="fig2")
    args = parser.parse_args()

    seq = collect_sequences(args.logs_root, args.scale, args.budget, args.methods)
    summary = summarize_sequences(seq)

    dimension = "40-D" if args.scale == "medium" else "80-D"
    seq.insert(3, "dimension", dimension)
    summary.insert(3, "dimension", dimension)

    args.results_root.mkdir(parents=True, exist_ok=True)
    seq_path = args.results_root / f"{args.figure_stem}_best_so_far_sequences_{dimension}.csv"
    summary_path = args.results_root / f"{args.figure_stem}_profile_convergence_{dimension}.csv"
    seq.to_csv(seq_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {seq_path}")
    print(f"Wrote {summary_path}")

    plot_fig2(summary, args.figures_root / args.figure_stem, args.methods)


if __name__ == "__main__":
    main()
