from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.problem.benchmark_generator import generate_instance, load_instance
from src.problem.sparse_reward_env import SparseRewardUAVEnv
from src.hsbo.hsbo_full import run_hsbo


ABLATIONS = {
    "hsbo_full": dict(),
    "hsbo_no_hierarchy": dict(use_hierarchy=False),
    "hsbo_no_presearch": dict(use_presearch=False),
    "hsbo_no_gradient": dict(use_gradient=False),
    "hsbo_no_pareto": dict(use_pareto=False),
    "hsbo_no_diversity": dict(use_diversity=False),
}


def get_instance(scale: str, seed: int):
    path = ROOT / "archive" / "data" / "datasets" / scale / f"instance_seed{seed}.json"
    if path.exists():
        return load_instance(path)
    return generate_instance(scale, seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", type=str, default="outputs/ablation")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    jobs = [(name, seed) for name in ABLATIONS for seed in args.seeds]
    for name, seed in tqdm(jobs, desc=f"{args.scale} ablations"):
        env = SparseRewardUAVEnv(get_instance(args.scale, seed))
        df = run_hsbo(env, budget=args.budget, seed=seed, method_name=name, **ABLATIONS[name])
        df.to_csv(out / f"{args.scale}_{name}_seed{seed}.csv", index=False)
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(out / f"{args.scale}_ablation_combined.csv", index=False)
    print(f"Saved ablation results to {out}")


if __name__ == "__main__":
    main()
