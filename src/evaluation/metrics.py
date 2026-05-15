from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass
class LogRow:
    iteration: int
    current_reward: float
    best_reward: float
    feasible: bool
    served_tasks: int
    best_served_tasks: int
    runtime_sec: float
    method: str
    seed: int
    scale: str


class ExperimentLogger:
    def __init__(self, method: str, seed: int, scale: str):
        self.method = method
        self.seed = seed
        self.scale = scale
        self.start = time.perf_counter()
        self.best = float("-inf")
        self.best_served_tasks = 0
        self.best_served_priority = 0.0
        self.rows: List[dict] = []
        self._seen_feasible = False

    def add(self, iteration: int, result) -> None:
        reward = float(result.reward)
        feasible = bool(result.feasible)
        first_feasible = feasible and not self._seen_feasible
        if feasible:
            self._seen_feasible = True
        details = getattr(result, "details", {}) or {}
        served_ratio = float(getattr(result, "served_ratio", details.get("served_ratio", 0.0)))
        served_priority = float(details.get("raw_packet_utility", details.get("served_priority", 0.0)))
        conflict_count = int(details.get("conflict_count", round(float(getattr(result, "conflict_penalty", 0.0)))))
        energy_cost = float(details.get("energy_cost", getattr(result, "energy_penalty", 0.0)))
        if reward > self.best:
            self.best = reward
            self.best_served_tasks = int(result.served_tasks)
            self.best_served_priority = served_priority
        row = {
            "eval_idx": int(iteration),
            "utility": reward,
            "served_count": int(result.served_tasks),
            "served_ratio": served_ratio,
            "served_priority": served_priority,
            "raw_packet_utility": served_priority,
            "success_flag": feasible,
            "first_feasible_flag": first_feasible,
            "window_feasible_attempts": int(details.get("window_feasible_attempts", 0)),
            "rate_feasible_attempts": int(details.get("rate_feasible_attempts", 0)),
            "transmission_feasible_attempts": int(details.get("transmission_feasible_attempts", 0)),
            "outage_count": int(details.get("outage_count", 0)),
            "duplicate_count": int(details.get("duplicate_count", 0)),
            "conflict_count": conflict_count,
            "energy_cost": energy_cost,
            "best_so_far": float(self.best),
            "runtime_sec": float(time.perf_counter() - self.start),
            "method": self.method,
            "seed": int(self.seed),
            "scale": self.scale,
            # Backward-compatible columns used by the original scripts.
            "iteration": int(iteration),
            "current_reward": reward,
            "best_reward": float(self.best),
            "feasible": feasible,
            "served_tasks": int(result.served_tasks),
            "best_served_tasks": int(self.best_served_tasks),
            "best_served_priority": float(self.best_served_priority),
        }
        for key in ("channel_model", "budget", "window_width", "rate_threshold_mbps"):
            if key in details:
                row[key] = details[key]
        self.rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize experiment logs.

    `feasible_eval_rate` is computed over all evaluations.
    `final_best_*` is computed from the last best-so-far value per seed.
    """
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["feasible"] = df["feasible"].astype(bool)
    if "best_served_tasks" not in df.columns:
        df["best_served_tasks"] = df["served_tasks"]
    final = df.sort_values("iteration").groupby(["scale", "method", "seed"], as_index=False).tail(1)
    final_summary = (
        final.groupby(["scale", "method"])
        .agg(
            final_best_mean=("best_reward", "mean"),
            final_best_std=("best_reward", "std"),
            final_best_max=("best_reward", "max"),
            final_best_served_mean=("best_served_tasks", "mean"),
            runtime_mean=("runtime_sec", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    eval_summary = (
        df.groupby(["scale", "method"])
        .agg(
            feasible_eval_rate=("feasible", "mean"),
            mean_current_reward=("current_reward", "mean"),
        )
        .reset_index()
    )
    return final_summary.merge(eval_summary, on=["scale", "method"], how="left")
