from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_METHODS = [
    'hsbo',
    'bo',
    'ga',
    'pso',
    'random',
    'greedy_distance',
    'greedy_reward_density',
]

LABEL_MAP = {
    'hsbo': 'HSBO',
    'bo': 'BO',
    'ga': 'GA',
    'pso': 'PSO',
    'random': 'Random',
    'greedy_distance': 'Greedy-distance',
    'greedy_reward_density': 'Greedy-reward-density',
}


def find_csv_files(result_dir: Path, method: str) -> List[Path]:
    files = []
    for p in result_dir.rglob('*.csv'):
        name = p.name.lower()
        if method in name:
            files.append(p)
    return sorted(files)


def get_reward_column(df: pd.DataFrame) -> str:
    candidates = [
        'reward', 'current_reward', 'objective', 'score', 'value',
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f'No reward-like column found. Columns = {list(df.columns)}')


def curve_from_file(csv_path: Path, budget: int) -> np.ndarray:
    df = pd.read_csv(csv_path)
    reward_col = get_reward_column(df)
    reward = df[reward_col].to_numpy(dtype=float)
    reward = reward[:budget]
    if reward.size == 0:
        reward = np.zeros(budget, dtype=float)
    best = np.maximum.accumulate(reward)
    if best.size < budget:
        pad_value = best[-1] if best.size else 0.0
        best = np.pad(best, (0, budget - best.size), constant_values=pad_value)
    return best


def load_method_curves(result_dir: Path, method: str, budget: int) -> Optional[np.ndarray]:
    files = find_csv_files(result_dir, method)
    if not files:
        return None
    curves = [curve_from_file(f, budget) for f in files]
    return np.vstack(curves)


def sem(arr: np.ndarray) -> np.ndarray:
    if arr.shape[0] <= 1:
        return np.zeros(arr.shape[1], dtype=float)
    return arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])


def plot_final(
    result_dir: Path,
    budget: int,
    out_path: Path,
    title: str,
    methods: List[str],
    collapse_zero_baselines: bool = True,
    ylabel: str = 'Best-so-far reward',
):
    x = np.arange(1, budget + 1)
    plt.figure(figsize=(7.4, 4.6))

    curves_map: Dict[str, np.ndarray] = {}
    for m in methods:
        curves = load_method_curves(result_dir, m, budget)
        if curves is not None:
            curves_map[m] = curves

    if 'hsbo' not in curves_map:
        raise FileNotFoundError('No HSBO per-seed CSV files found in the result directory.')

    hsbo_curves = curves_map['hsbo']
    hsbo_mean = hsbo_curves.mean(axis=0)
    hsbo_se = sem(hsbo_curves)
    plt.plot(x, hsbo_mean, linewidth=2.4, label='HSBO')
    plt.fill_between(x, hsbo_mean - hsbo_se, hsbo_mean + hsbo_se, alpha=0.18)

    baseline_methods = [m for m in methods if m != 'hsbo' and m in curves_map]
    if baseline_methods:
        baseline_max = max(float(curves_map[m].max()) for m in baseline_methods)
        if collapse_zero_baselines and baseline_max == 0.0:
            plt.plot(x, np.zeros_like(x), '--', linewidth=1.8, label='All baselines')
        else:
            for m in baseline_methods:
                mean_curve = curves_map[m].mean(axis=0)
                plt.plot(x, mean_curve, linewidth=1.3, label=LABEL_MAP.get(m, m))

    plt.xlabel('Number of objective evaluations')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linewidth=0.4, alpha=0.35)
    plt.xlim(1, budget)
    plt.legend(frameon=False, loc='best')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Redraw paper-ready convergence figures.')
    parser.add_argument('--medium-dir', type=str, required=True, help='Directory containing per-seed CSV logs for the 40-D experiment.')
    parser.add_argument('--large-dir', type=str, required=True, help='Directory containing per-seed CSV logs for the 80-D experiment.')
    parser.add_argument('--medium-budget', type=int, default=300)
    parser.add_argument('--large-budget', type=int, default=300)
    parser.add_argument('--out-medium', type=str, default='paper_convergence_medium_final.png')
    parser.add_argument('--out-large', type=str, default='paper_convergence_large_final.png')
    args = parser.parse_args()

    plot_final(
        result_dir=Path(args.medium_dir),
        budget=args.medium_budget,
        out_path=Path(args.out_medium),
        title='Convergence on the 40-D instance',
        methods=DEFAULT_METHODS,
        collapse_zero_baselines=True,
    )
    plot_final(
        result_dir=Path(args.large_dir),
        budget=args.large_budget,
        out_path=Path(args.out_large),
        title='Convergence on the 80-D stress-test instance',
        methods=DEFAULT_METHODS,
        collapse_zero_baselines=True,
    )
    print('Saved:', args.out_medium)
    print('Saved:', args.out_large)
