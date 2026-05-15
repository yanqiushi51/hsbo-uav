#!/usr/bin/env python3
"""
Run enhanced baselines (CMA-ES, DE, and their repaired variants).

Usage:
  python scripts/run_enhanced_baselines.py \
    --scale medium --methods cmaes cmaes_repair de de_repair \
    --budget 300 --seeds 0 1 2 3 4 --out results/enhanced_medium
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from src.problem.benchmark_generator import generate_instance, load_instance
from src.problem.sparse_reward_env import SparseRewardUAVEnv
from src.baselines import (
    run_cmaes, run_cmaes_repair, run_cmaes_seeded, run_cmaes_repair_seeded,
    run_de, run_de_repair, run_de_seeded, run_de_repair_seeded,
)

ENHANCED_METHODS = {
    "cmaes": run_cmaes,
    "cmaes_repair": run_cmaes_repair,
    "cmaes_seeded": run_cmaes_seeded,
    "cmaes_repair_seeded": run_cmaes_repair_seeded,
    "de": run_de,
    "de_repair": run_de_repair,
    "de_seeded": run_de_seeded,
    "de_repair_seeded": run_de_repair_seeded,
}


def get_instance(scale: str, seed: int):
    path = ROOT / "datasets" / scale / f"instance_seed{seed}.json"
    if path.exists():
        return load_instance(path)
    return generate_instance(scale, seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--methods", nargs="+",
                        default=["cmaes", "cmaes_repair", "de", "de_repair"])
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", type=str, default="results/enhanced")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        if method not in ENHANCED_METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(ENHANCED_METHODS)}")

    jobs = [(m, s) for m in args.methods for s in args.seeds]
    all_frames = []

    for method, seed in tqdm(jobs, desc=f"{args.scale} enhanced"):
        instance = get_instance(args.scale, seed)
        env = SparseRewardUAVEnv(instance)
        df = ENHANCED_METHODS[method](env, budget=args.budget, seed=seed)
        path = out / f"{args.scale}_{method}_seed{seed}.csv"
        df.to_csv(path, index=False)
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    combined.to_csv(out / f"{args.scale}_combined.csv", index=False)
    print(f"Saved {len(all_frames)} results to {out}")


if __name__ == "__main__":
    main()
