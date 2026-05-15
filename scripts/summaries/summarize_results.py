from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import summarize_dataframe


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
    summary = summarize_dataframe(df)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(f"Saved summary to {out}")


if __name__ == "__main__":
    main()
