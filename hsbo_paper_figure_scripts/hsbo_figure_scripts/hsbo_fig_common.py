#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utilities for HSBO paper figures.

Assumptions:
- Each per-seed CSV contains one evaluation trajectory.
- Reward column is one of: reward, current_reward, objective, score, value.
- Method is inferred from a 'method' column if present; otherwise from the file name.
- Summary CSVs are ignored for per-seed plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import re

import numpy as np
import pandas as pd


METHOD_ORDER = [
    "hsbo",
    "greedy_reward_density",
    "pso",
    "greedy_distance",
    "random",
    "ga",
    "bo",
]

METHOD_LABEL = {
    "hsbo": "HSBO",
    "greedy_reward_density": "G-RD",
    "greedy_distance": "G-D",
    "random": "RS",
    "ga": "GA",
    "pso": "PSO",
    "bo": "BO",
}

# JSEE publication colour scheme: HSBO deep blue, baselines grayscale
METHOD_COLOR = {
    "hsbo": "#2166AC",
    "greedy_reward_density": "#7F7F7F",
    "pso": "#999999",
    "greedy_distance": "#B0B0B0",
    "random": "#C0C0C0",
    "ga": "#D0D0D0",
    "bo": "#E0E0E0",
}

BASELINE_METHODS = ["greedy_reward_density", "pso", "greedy_distance", "random", "ga", "bo"]

REWARD_COLS = ["reward", "current_reward", "objective", "score", "value"]


@dataclass
class RunRecord:
    scale: str
    method: str
    seed: str
    final_best: float
    feasible_rate: float
    first_feasible: Optional[int]
    runtime: Optional[float] = None


def _method_from_name(path: Path) -> Optional[str]:
    name = path.stem.lower()
    for method in sorted(METHOD_ORDER, key=len, reverse=True):
        if re.search(rf"(^|[_\-]){re.escape(method)}([_\-]|\d|$)", name):
            return method
    for method in sorted(METHOD_ORDER, key=len, reverse=True):
        if method in name:
            return method
    return None


def _seed_from_name(path: Path) -> str:
    name = path.stem.lower()
    m = re.search(r"seed[_\-]?(\d+)", name)
    if m:
        return m.group(1)
    return path.stem


def reward_column(df: pd.DataFrame) -> str:
    for col in REWARD_COLS:
        if col in df.columns:
            return col
    raise ValueError(f"No reward column found. Available columns: {list(df.columns)}")


def iter_per_seed_csvs(result_dir: Path) -> Iterable[Path]:
    for path in sorted(result_dir.rglob("*.csv")):
        lower = path.name.lower()
        if "summary" in lower or "combined" in lower or "aggregate" in lower:
            continue
        yield path


def load_records(result_dir: str | Path, scale: str, budget: Optional[int] = None) -> pd.DataFrame:
    """Load per-seed trajectories and summarize final-best statistics."""
    result_dir = Path(result_dir)
    records: List[RunRecord] = []

    csvs = list(iter_per_seed_csvs(result_dir))
    if not csvs:
        raise FileNotFoundError(
            f"No per-seed CSV files found under {result_dir}. "
            "Do not point this script to summary-only folders."
        )

    for path in csvs:
        df = pd.read_csv(path)
        method = None
        if "method" in df.columns and len(df["method"].dropna()) > 0:
            method = str(df["method"].dropna().iloc[0]).lower()
        if method is None:
            method = _method_from_name(path)
        if method is None:
            continue

        rcol = reward_column(df)
        rewards = df[rcol].to_numpy(dtype=float)
        if budget is not None:
            rewards = rewards[:budget]
        if rewards.size == 0:
            continue

        best = np.maximum.accumulate(rewards)
        positive = rewards > 0
        first_feasible = int(np.argmax(positive) + 1) if positive.any() else None

        runtime = None
        for runtime_col in ["runtime", "elapsed", "time"]:
            if runtime_col in df.columns and len(df[runtime_col].dropna()) > 0:
                try:
                    runtime = float(df[runtime_col].dropna().iloc[-1])
                except Exception:
                    runtime = None
                break

        records.append(
            RunRecord(
                scale=scale,
                method=method,
                seed=_seed_from_name(path),
                final_best=float(best[-1]),
                feasible_rate=float(positive.mean()),
                first_feasible=first_feasible,
                runtime=runtime,
            )
        )

    if not records:
        raise RuntimeError(f"CSV files were found in {result_dir}, but no known methods could be parsed.")

    return pd.DataFrame([r.__dict__ for r in records])


def load_multiscale(
    small_dir: str | Path,
    medium_dir: str | Path,
    large_dir: str | Path,
    small_budget: Optional[int] = None,
    medium_budget: Optional[int] = None,
    large_budget: Optional[int] = None,
) -> pd.DataFrame:
    dfs = [
        load_records(small_dir, "18-D", small_budget),
        load_records(medium_dir, "40-D", medium_budget),
        load_records(large_dir, "80-D", large_budget),
    ]
    return pd.concat(dfs, ignore_index=True)


def method_sort_key(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


def ordered_methods(df: pd.DataFrame) -> List[str]:
    return sorted(df["method"].unique().tolist(), key=method_sort_key)


def label(method: str) -> str:
    return METHOD_LABEL.get(method, method)
