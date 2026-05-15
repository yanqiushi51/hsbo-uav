from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.problem.benchmark_generator import generate_instance, load_instance
from src.problem.sparse_reward_env import SparseRewardUAVEnv
from src.baselines import run_random_search, run_greedy, run_greedy_distance, run_greedy_reward_density, run_ga, run_pso, run_standard_bo
from src.hsbo.hsbo_full import run_hsbo


METHODS = {
    "random": run_random_search,
    "greedy": run_greedy,
    "greedy_distance": run_greedy_distance,
    "greedy_reward_density": run_greedy_reward_density,
    "ga": run_ga,
    "pso": run_pso,
    "bo": run_standard_bo,
    "hsbo": run_hsbo,
}


def get_instance(scale: str, seed: int):
    path = ROOT / "datasets" / scale / f"instance_seed{seed}.json"
    if path.exists():
        return load_instance(path)
    return generate_instance(scale, seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--methods", nargs="+", default=["random", "greedy", "ga", "pso", "bo", "hsbo"])
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", type=str, default="results/main")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_frames = []
    jobs = [(m, s) for m in args.methods for s in args.seeds]
    for method, seed in tqdm(jobs, desc=f"{args.scale} experiments"):
        if method not in METHODS:
            raise ValueError(f"Unknown method {method}; choose from {list(METHODS)}")
        instance = get_instance(args.scale, seed)
        env = SparseRewardUAVEnv(instance)
        df = METHODS[method](env, budget=args.budget, seed=seed)
        path = out / f"{args.scale}_{method}_seed{seed}.csv"
        df.to_csv(path, index=False)
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    combined.to_csv(out / f"{args.scale}_combined.csv", index=False)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
