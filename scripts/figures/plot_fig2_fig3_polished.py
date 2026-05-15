from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "archive" / "experiments" / "results"
FIGURES = ROOT / "paper" / "figures"


METHOD_LABELS = {
    "ft_hsbo": "FT-HSBO",
    "max_rate_greedy": "Max-Rate",
    "edf_rate_greedy": "EDF",
    "nearest_feasible_user": "Nearest",
    "de_rp": "DE-rp",
    "repair_bo": "Repair BO",
    "full_space_bo": "Full BO",
    "rs": "RS",
}

DISCOVERY_ORDER = [
    "ft_hsbo",
    "max_rate_greedy",
    "edf_rate_greedy",
    "nearest_feasible_user",
    "de_rp",
    "repair_bo",
    "full_space_bo",
    "rs",
]

SENSITIVITY_ORDER = [
    "ft_hsbo",
    "max_rate_greedy",
    "edf_rate_greedy",
    "de_rp",
    "full_space_bo",
]

COLORS = {
    "40-D": "#A1BCE1",
    "80-D": "#F0C47E",
    "ft_hsbo": "#A1BCE1",
    "max_rate_greedy": "#F0C47E",
    "edf_rate_greedy": "#A3CBA9",
    "de_rp": "#E5A49F",
    "full_space_bo": "#BDB4D8",
}

HATCHES = {
    "40-D": "",
    "80-D": "///",
}

FIG2_ORDER = [
    "ft_hsbo",
    "max_rate_greedy",
    "de_rp",
    "repair_bo",
    "full_space_bo",
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".png", ".svg"]:
        out = FIGURES / f"{stem}{suffix}"
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 420
        fig.savefig(out, **kwargs)
        print(f"Wrote {out}")


def method_name(method_id: str) -> str:
    return METHOD_LABELS.get(method_id, method_id)


def plot_fig2() -> None:
    df = pd.read_csv(RESULTS / "main_comparison_discovery_selected.csv")
    df = df[df["method_id"].isin(FIG2_ORDER)].copy()
    df["method_id"] = pd.Categorical(df["method_id"], FIG2_ORDER, ordered=True)
    df["dimension"] = pd.Categorical(df["dimension"], ["40-D", "80-D"], ordered=True)
    df = df.sort_values(["method_id", "dimension"])
    df.to_csv(RESULTS / "fig2_discovery_plot_values.csv", index=False)

    methods = FIG2_ORDER
    x = np.arange(len(methods))
    width = 0.36
    offsets = {"40-D": -width / 2, "80-D": width / 2}

    fig, ax = plt.subplots(figsize=(4.35, 2.55))
    for dim in ["40-D", "80-D"]:
        sub = df[df["dimension"] == dim].set_index("method_id")
        vals = [float(sub.loc[m, "conditional_median_first_feasible_index"]) for m in methods]
        bars = ax.bar(
            x + offsets[dim],
            vals,
            width,
            label=dim,
            color=COLORS[dim],
            edgecolor="#555555",
            linewidth=0.55,
        )
        for bar, val in zip(bars, vals):
            if val > 1.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val * 1.08,
                    f"{val:g}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )
    ax.set_yscale("log")
    ax.set_ylim(0.8, 220)
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100", "200"])
    ax.set_ylabel("Conditional median first feasible\nindex (log)")
    ax.set_xticks(x)
    ax.set_xticklabels([method_name(m) for m in methods], rotation=0, ha="center")
    ax.grid(axis="y", which="both", color="#d7dce2", linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncols=2, handlelength=1.4, columnspacing=1.2)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.965, bottom=0.18)

    save_all(fig, "fig2_first_feasible_wcl")
    plt.close(fig)


def plot_fig2_compact() -> None:
    df = pd.read_csv(RESULTS / "main_comparison_discovery_selected.csv")
    df = df[df["method_id"].isin(FIG2_ORDER)].copy()
    df["method_id"] = pd.Categorical(df["method_id"], FIG2_ORDER, ordered=True)
    df["dimension"] = pd.Categorical(df["dimension"], ["40-D", "80-D"], ordered=True)
    df = df.sort_values(["method_id", "dimension"])
    df.to_csv(RESULTS / "fig2_discovery_compact_plot_values.csv", index=False)

    methods = FIG2_ORDER
    x = np.arange(len(methods))
    width = 0.36
    offsets = {"40-D": -width / 2, "80-D": width / 2}

    fig, ax = plt.subplots(figsize=(3.55, 2.45))
    for dim in ["40-D", "80-D"]:
        sub = df[df["dimension"] == dim].set_index("method_id")
        vals = [float(sub.loc[m, "conditional_median_first_feasible_index"]) for m in methods]
        bars = ax.bar(
            x + offsets[dim],
            vals,
            width,
            label=dim,
            color="#d9e8f5" if dim == "40-D" else "#f3d8a8",
            edgecolor="#333333",
            linewidth=0.5,
            hatch=HATCHES[dim],
        )
        for bar, val in zip(bars, vals):
            if val > 1.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val * 1.12,
                    f"{val:g}",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    rotation=90,
                )
    ax.set_yscale("log")
    ax.set_ylim(0.8, 260)
    ax.set_yticks([1, 3, 10, 30, 100, 300])
    ax.set_yticklabels(["1", "3", "10", "30", "100", "300"])
    ax.set_ylabel("Conditional median FFI")
    labels = []
    for method in methods:
        sub = df[df["method_id"] == method].set_index("dimension")
        sr40 = float(sub.loc["40-D", "discovery_success_rate"])
        sr80 = float(sub.loc["80-D", "discovery_success_rate"])
        labels.append(f"{method_name(method)}\nSR {sr40:.2g}/{sr80:.2g}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=5.8)
    ax.grid(axis="y", which="both", color="#d7dce2", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncols=2, handlelength=1.3)
    ax.text(
        0.98,
        0.96,
        "20 seeds\nsuccessful runs only",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.2},
    )
    save_all(fig, "fig2_discovery_compact")
    plt.close(fig)


def sensitivity_values(path: Path, varied_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["method"].isin(SENSITIVITY_ORDER)].copy()
    df["method"] = pd.Categorical(df["method"], SENSITIVITY_ORDER, ordered=True)
    df = df.sort_values(["method", varied_col])
    return df


def plot_sensitivity_panel(ax: plt.Axes, df: pd.DataFrame, xcol: str, title: str, xlabel: str) -> None:
    x_values = sorted(df[xcol].astype(float).unique())
    x = np.arange(len(x_values))
    width = 0.15
    offsets = np.linspace(-2, 2, len(SENSITIVITY_ORDER)) * width
    for offset, method in zip(offsets, SENSITIVITY_ORDER):
        g = df[df["method"] == method].sort_values(xcol)
        y = g["mean_utility"].astype(float).to_numpy()
        ax.bar(
            x + offset,
            y,
            label=method_name(method),
            color=COLORS[method],
            width=width,
            edgecolor="#555555",
            linewidth=0.5,
        )
    ax.text(0.02, 0.96, title, transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean utility")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x_values])
    ax.grid(axis="y", color="#d7dce2", linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)


def plot_fig3() -> None:
    window = sensitivity_values(RESULTS / "summary_sensitivity_window.csv", "window_width")
    rate = sensitivity_values(RESULTS / "summary_sensitivity_rate.csv", "rate_threshold_mbps")
    window.to_csv(RESULTS / "fig3_sensitivity_window_plot_values.csv", index=False)
    rate.to_csv(RESULTS / "fig3_sensitivity_rate_plot_values.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(3.55, 4.55), sharey=True)
    plot_sensitivity_panel(axes[0], window, "window_width", "(a)", "Window width (s)")
    axes[0].set_ylim(0, 120)

    plot_sensitivity_panel(axes[1], rate, "rate_threshold_mbps", "(b)", r"Rate threshold $R_{\min}$ (Mbps)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        ncols=3,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.1,
        handlelength=1.2,
    )
    fig.subplots_adjust(top=0.88, hspace=0.42)

    save_all(fig, "fig3_sensitivity_wcl")
    plt.close(fig)


def plot_fig3_compact() -> None:
    window = sensitivity_values(RESULTS / "summary_sensitivity_window.csv", "window_width")
    rate = sensitivity_values(RESULTS / "summary_sensitivity_rate.csv", "rate_threshold_mbps")

    fig, axes = plt.subplots(2, 1, figsize=(3.55, 4.2), sharey=False)
    plot_sensitivity_panel(axes[0], window, "window_width", "(a) Window width", "Window width (s)")
    axes[0].set_xticks([15, 30, 60])
    axes[0].set_ylim(0, 62)
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.26), ncols=3, handlelength=1.4)

    plot_sensitivity_panel(axes[1], rate, "rate_threshold_mbps", "(b) Rate threshold", r"$R_{\min}$ (Mbps)")
    axes[1].set_xticks([8, 12, 15])
    axes[1].set_ylim(0, 140)
    legend = axes[1].get_legend()
    if legend:
        legend.remove()
    axes[1].text(
        0.02,
        0.96,
        "10-seed diagnostic tests, 40-D",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.2},
    )
    fig.subplots_adjust(hspace=0.45)
    save_all(fig, "fig3_sensitivity_compact")
    plt.close(fig)


def main() -> None:
    setup_style()
    plot_fig2()
    plot_fig2_compact()
    plot_fig3()
    plot_fig3_compact()


if __name__ == "__main__":
    main()
