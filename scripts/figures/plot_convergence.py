from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    result_dir = Path(args.results)
    frames = []
    for p in result_dir.glob("*.csv"):
        if "combined" in p.name or "summary" in p.name:
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit(f"No result CSV files found in {result_dir}")
    df = pd.concat(frames, ignore_index=True)
    grouped = df.groupby(["method", "iteration"], as_index=False).agg(mean_best=("best_reward", "mean"), std_best=("best_reward", "std"))

    plt.figure(figsize=(8, 5))
    for method, g in grouped.groupby("method"):
        g = g.sort_values("iteration")
        plt.plot(g["iteration"], g["mean_best"], label=method)
    plt.xlabel("Evaluation budget")
    plt.ylabel("Best-so-far reward")
    plt.title("Convergence comparison")
    plt.legend()
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
