from __future__ import annotations

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


SUMMARY_TARGETS = {
    "main": ("main", "summary_main.csv"),
    "ablation": ("ablation", "summary_ablation.csv"),
    "window": ("window_", "summary_sensitivity_window.csv"),
    "rate": ("rate_", "summary_sensitivity_rate.csv"),
    "channel": ("channel_", "summary_channel.csv"),
    "budget": ("budget_", "summary_budget.csv"),
}


def read_logs(logs_root: Path, prefix: str) -> pd.DataFrame:
    frames = []
    for path in logs_root.glob("**/seed_*.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "experiment" not in df.columns:
            continue
        experiment = str(df["experiment"].iloc[0])
        if experiment == prefix or experiment.startswith(prefix):
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    defaults = {
        "channel_model": "distance",
        "budget": np.nan,
        "window_width": np.nan,
        "rate_threshold_mbps": np.nan,
        "success_flag": False,
        "served_count": 0,
        "served_ratio": 0.0,
        "outage_count": 0,
        "window_feasible_attempts": 0,
        "duplicate_count": 0,
        "conflict_count": 0,
        "energy_cost": 0.0,
    }
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
    utility = df["utility"].astype(float)
    energy = df["energy_cost"].astype(float)
    duplicate = df["duplicate_count"].astype(float)
    conflict = df["conflict_count"].astype(float)
    served_count = df["served_count"].astype(float)
    estimated_raw_utility = utility + 0.01 * energy + 2.0 * (duplicate + conflict)
    fallback_raw_utility = np.where(
        served_count <= 0.0,
        0.0,
        np.where(utility > 0.0, estimated_raw_utility, np.nan),
    )
    if "raw_packet_utility" not in df.columns:
        df["raw_packet_utility"] = np.nan
    if "served_priority" in df.columns:
        df["raw_packet_utility"] = df["raw_packet_utility"].fillna(df["served_priority"])
    df["raw_packet_utility"] = df["raw_packet_utility"].fillna(pd.Series(fallback_raw_utility, index=df.index))
    df["success_flag"] = df["success_flag"].astype(bool)
    group_cols = [
        "experiment",
        "scale",
        "channel_model",
        "method",
        "budget",
        "window_width",
        "rate_threshold_mbps",
    ]
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        per_seed = []
        best_rows = []
        first_feasible = []
        for seed, gs in g.groupby("seed"):
            gs = gs.sort_values("eval_idx")
            best_value = float(gs["best_so_far"].iloc[-1]) if "best_so_far" in gs.columns else float(gs["utility"].max())
            success = bool(gs["success_flag"].any())
            ff = float(gs.loc[gs["success_flag"], "eval_idx"].iloc[0]) if success else np.nan
            idx = gs["utility"].idxmax()
            best_rows.append(gs.loc[idx])
            first_feasible.append(ff)
            per_seed.append({"seed": seed, "best": best_value, "success": success})
        seed_df = pd.DataFrame(per_seed)
        best_df = pd.DataFrame(best_rows)
        outage_den = g["window_feasible_attempts"].replace(0, np.nan)
        outage_ratio = (g["outage_count"] / outage_den).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "num_seeds": int(seed_df["seed"].nunique()),
                "mean_utility": float(seed_df["best"].mean()),
                "std_utility": float(seed_df["best"].std(ddof=1)) if len(seed_df) > 1 else 0.0,
                "median_utility": float(seed_df["best"].median()),
                "max_utility": float(seed_df["best"].max()),
                "success_rate": float(seed_df["success"].mean()),
                "mean_first_feasible": float(np.nanmean(first_feasible)) if np.isfinite(first_feasible).any() else np.nan,
                "median_first_feasible": float(np.nanmedian(first_feasible)) if np.isfinite(first_feasible).any() else np.nan,
                "feasible_eval_rate": float(g["success_flag"].mean()),
                "mean_served_priority": float(best_df["raw_packet_utility"].mean()) if "raw_packet_utility" in best_df else 0.0,
                "std_served_priority": float(best_df["raw_packet_utility"].std(ddof=1)) if "raw_packet_utility" in best_df and len(best_df) > 1 else 0.0,
                "mean_served_ratio": float(best_df["served_ratio"].mean()) if "served_ratio" in best_df else 0.0,
                "std_served_ratio": float(best_df["served_ratio"].std(ddof=1)) if "served_ratio" in best_df and len(best_df) > 1 else 0.0,
                "mean_outage_ratio": float(outage_ratio.mean()),
                "mean_energy": float(best_df["energy_cost"].mean()) if "energy_cost" in best_df else 0.0,
            }
        )
        rows.append(row)
    required_order = [
        "experiment",
        "scale",
        "channel_model",
        "method",
        "budget",
        "num_seeds",
        "mean_utility",
        "std_utility",
        "median_utility",
        "max_utility",
        "success_rate",
        "mean_first_feasible",
        "median_first_feasible",
        "feasible_eval_rate",
        "mean_served_priority",
        "std_served_priority",
        "mean_served_ratio",
        "std_served_ratio",
        "mean_outage_ratio",
        "mean_energy",
        "window_width",
        "rate_threshold_mbps",
    ]
    return pd.DataFrame(rows).sort_values(["experiment", "scale", "method"])[required_order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", default="outputs/logs")
    parser.add_argument("--results-root", default="outputs/results")
    parser.add_argument("--which", choices=["all", *SUMMARY_TARGETS.keys()], default="all")
    args = parser.parse_args()

    logs_root = ROOT / args.logs_root
    results_root = ROOT / args.results_root
    targets = SUMMARY_TARGETS if args.which == "all" else {args.which: SUMMARY_TARGETS[args.which]}
    results_root.mkdir(parents=True, exist_ok=True)
    for name, (prefix, filename) in targets.items():
        df = read_logs(logs_root, prefix)
        summary = summarize(df)
        out = results_root / filename
        summary.to_csv(out, index=False)
        print(f"Wrote {out} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
