from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from .benchmark_generator import BenchmarkInstance


@dataclass
class EvaluationResult:
    reward: float
    feasible: bool
    served_tasks: int
    total_travel: float
    conflict_penalty: float
    delay_penalty: float
    energy_penalty: float
    details: Dict


class SparseRewardUAVEnv:
    """Continuous-vector evaluator for sparse-reward multi-UAV scheduling.

    Vector structure:
    - upper variables: for each UAV, speed_norm and heading_norm, both in [0, 1]
    - lower variables: for each UAV and action slot, task_key and proposed_start_norm

    Version 2 adds cooperative tasks and narrower time windows. A cooperative task
    only yields reward when the required number of distinct UAVs serve it within
    a synchronization tolerance. This makes the reward sparse and coupled.
    """

    def __init__(self, instance: BenchmarkInstance):
        self.instance = instance
        self.n = instance.n_uavs
        self.m = instance.n_tasks
        self.k = instance.max_actions_per_uav
        self.dim_upper = 2 * self.n
        self.dim_lower = 2 * self.n * self.k
        self.dim = self.dim_upper + self.dim_lower
        self.horizon = instance.horizon

        self.task_xy = np.array([[t.x, t.y] for t in instance.tasks], dtype=float)
        self.task_rewards = np.array([t.reward for t in instance.tasks], dtype=float)
        self.task_service = np.array([t.service_time for t in instance.tasks], dtype=float)
        self.task_earliest = np.array([t.earliest for t in instance.tasks], dtype=float)
        self.task_latest = np.array([t.latest for t in instance.tasks], dtype=float)
        self.task_risk = np.array([t.risk for t in instance.tasks], dtype=float)
        self.task_type = np.array([t.task_type for t in instance.tasks], dtype=object)
        self.required_uavs = np.array([t.required_uavs for t in instance.tasks], dtype=int)
        self.sync_tolerance = np.array([t.sync_tolerance for t in instance.tasks], dtype=float)
        self.uav_xy = np.array([[u.x, u.y] for u in instance.uavs], dtype=float)
        self.v_min = np.array([u.v_min for u in instance.uavs], dtype=float)
        self.v_max = np.array([u.v_max for u in instance.uavs], dtype=float)

        self.energy_weight = 0.010
        self.delay_weight = 0.075
        self.duplicate_penalty = 5.0
        self.failed_action_penalty = 0.50
        self.overtime_penalty = 8.0
        self.conflict_time_threshold = 2.5
        self.conflict_penalty_weight = 4.0

    def random_vector(self, rng: np.random.Generator) -> np.ndarray:
        return rng.random(self.dim)

    def split(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {x.shape[0]}")
        x = np.clip(x, 0.0, 1.0)
        return x[: self.dim_upper], x[self.dim_upper :]

    def decode_upper(self, upper: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        upper = np.clip(np.asarray(upper, dtype=float), 0.0, 1.0)
        speeds = self.v_min + upper[0::2] * (self.v_max - self.v_min)
        headings = upper[1::2] * 2.0 * np.pi
        return speeds, headings

    def decode_lower(self, lower: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.clip(np.asarray(lower, dtype=float), 0.0, 1.0)
        task_keys = lower[0::2].reshape(self.n, self.k)
        start_norm = lower[1::2].reshape(self.n, self.k)
        task_ids = np.minimum((task_keys * self.m).astype(int), self.m - 1)
        proposed_starts = start_norm * self.horizon
        return task_ids, proposed_starts

    def _simulate_attempts(self, x: np.ndarray):
        upper, lower = self.split(x)
        speeds, headings = self.decode_upper(upper)
        task_ids, proposed_starts = self.decode_lower(lower)
        attempts_by_task: Dict[int, List[Dict]] = {j: [] for j in range(self.m)}
        total_travel = 0.0
        delay_penalty = 0.0
        overtime_count = 0
        route_details = []

        for i in range(self.n):
            pos = self.uav_xy[i].copy()
            current_time = 0.0
            route = []
            for slot in range(self.k):
                j = int(task_ids[i, slot])
                task_pos = self.task_xy[j]
                vec = task_pos - pos
                dist = float(np.linalg.norm(vec))
                if dist <= 1e-9:
                    bearing = headings[i]
                else:
                    bearing = float(np.arctan2(vec[1], vec[0]) % (2.0 * np.pi))
                diff = abs(np.angle(np.exp(1j * (bearing - headings[i]))))
                heading_factor = max(0.30, 0.65 + 0.35 * np.cos(diff))
                effective_speed = max(1e-6, speeds[i] * heading_factor)
                travel_time = dist / effective_speed
                arrival = current_time + travel_time
                proposed = float(proposed_starts[i, slot])
                # In v2 the lower-level timing variable is a true scheduled
                # service start. A task scores only if this scheduled time lies
                # inside the task window and the UAV can physically arrive by then.
                # This makes random timing sparse and prevents simple route-only
                # greedy heuristics from becoming near-oracle solvers.
                start = proposed
                finish = start + self.task_service[j]

                valid_window = (arrival <= start) and (self.task_earliest[j] <= start <= self.task_latest[j]) and (finish <= self.horizon)
                if start > self.task_latest[j]:
                    delay_penalty += start - self.task_latest[j]
                elif start < self.task_earliest[j]:
                    delay_penalty += self.task_earliest[j] - start
                elif arrival > start:
                    delay_penalty += arrival - start
                if finish > self.horizon:
                    overtime_count += 1

                attempt = dict(
                    task=j,
                    uav=i,
                    slot=slot,
                    start=float(start),
                    finish=float(finish),
                    arrival=float(arrival),
                    dist=float(dist),
                    valid_window=bool(valid_window),
                    type=str(self.task_type[j]),
                    used=False,
                    reason="candidate" if valid_window else "invalid_time",
                )
                attempts_by_task[j].append(attempt)
                total_travel += dist

                # All actions consume travel and time, even when they eventually do not score.
                current_time = min(max(finish if valid_window else arrival, current_time), self.horizon)
                pos = task_pos.copy()
                route.append(attempt.copy())
            route_details.append(route)
        return attempts_by_task, total_travel, delay_penalty, overtime_count, route_details

    def evaluate(self, x: np.ndarray) -> EvaluationResult:
        attempts_by_task, total_travel, delay_penalty, overtime_count, route_details = self._simulate_attempts(x)
        total_reward = 0.0
        conflict_penalty = 0.0
        duplicate_count = 0
        failed_count = 0
        served_global = set()
        used_attempts = []

        # Score each task. Ordinary/critical tasks need one valid attempt. Cooperative
        # tasks need multiple distinct UAVs with synchronized start times.
        for j, attempts in attempts_by_task.items():
            valid = [a for a in attempts if a["valid_window"]]
            if not valid:
                failed_count += len(attempts)
                continue

            if self.task_type[j] == "cooperative":
                req = int(self.required_uavs[j])
                tau = float(self.sync_tolerance[j])
                best_group = None
                valid_sorted = sorted(valid, key=lambda a: a["start"])
                for idx, center_attempt in enumerate(valid_sorted):
                    s0 = center_attempt["start"]
                    group = []
                    used_uavs = set()
                    for a in valid_sorted:
                        if abs(a["start"] - s0) <= tau and a["uav"] not in used_uavs:
                            group.append(a)
                            used_uavs.add(a["uav"])
                    if len(group) >= req:
                        group = sorted(group, key=lambda a: abs(a["start"] - s0))[:req]
                        if best_group is None or np.var([g["start"] for g in group]) < np.var([g["start"] for g in best_group]):
                            best_group = group
                if best_group is None:
                    # Strong sparse-reward behavior: attempts to a cooperative task
                    # are useless unless enough UAVs synchronize.
                    failed_count += len(valid)
                    continue
                mean_start = float(np.mean([g["start"] for g in best_group]))
                sync_bonus = 1.0 + 0.08 * max(0.0, (tau - np.std([g["start"] for g in best_group])) / max(tau, 1e-6))
                risk_bonus = 1.0 + 0.15 * self.task_risk[j]
                early_bonus = 1.0 + 0.03 * max(0.0, (self.task_latest[j] - mean_start) / max(1.0, self.horizon))
                total_reward += self.task_rewards[j] * risk_bonus * early_bonus * sync_bonus
                served_global.add(j)
                for a in best_group:
                    a["used"] = True
                    a["reason"] = "served_cooperative"
                    used_attempts.append(a)
                extra = max(0, len(valid) - len(best_group))
                duplicate_count += extra
            else:
                # Ordinary/critical: choose the best single valid attempt. Critical
                # tasks have narrower windows from the generator, so no special case
                # is needed here.
                best = min(valid, key=lambda a: (a["start"], a["dist"]))
                risk_bonus = 1.0 + 0.15 * self.task_risk[j]
                early_bonus = 1.0 + 0.05 * max(0.0, (self.task_latest[j] - best["start"]) / max(1.0, self.horizon))
                total_reward += self.task_rewards[j] * risk_bonus * early_bonus
                served_global.add(j)
                best["used"] = True
                best["reason"] = "served"
                used_attempts.append(best)
                duplicate_count += max(0, len(valid) - 1)

        # Cross-UAV temporal conflict on actually scored attempts.
        used_sorted = sorted(used_attempts, key=lambda a: a["start"])
        for a_idx in range(len(used_sorted) - 1):
            a = used_sorted[a_idx]
            b = used_sorted[a_idx + 1]
            if a["uav"] != b["uav"] and abs(a["start"] - b["start"]) < self.conflict_time_threshold:
                # Cooperative tasks are allowed to synchronize on the same task.
                if not (a["task"] == b["task"] and self.task_type[a["task"]] == "cooperative"):
                    conflict_penalty += self.conflict_penalty_weight

        energy_penalty = self.energy_weight * total_travel
        duplicate_penalty = self.duplicate_penalty * duplicate_count
        overtime_penalty = self.overtime_penalty * overtime_count
        failed_penalty = self.failed_action_penalty * failed_count
        total = (
            total_reward
            - energy_penalty
            - self.delay_weight * delay_penalty
            - duplicate_penalty
            - overtime_penalty
            - conflict_penalty
            - failed_penalty
        )
        reward = float(max(0.0, total))
        feasible = len(served_global) > 0 and reward > 0.0
        return EvaluationResult(
            reward=reward,
            feasible=feasible,
            served_tasks=len(served_global),
            total_travel=float(total_travel),
            conflict_penalty=float(conflict_penalty),
            delay_penalty=float(delay_penalty),
            energy_penalty=float(energy_penalty),
            details={
                "routes": route_details,
                "served_global": sorted(served_global),
                "duplicate_count": duplicate_count,
                "overtime_count": overtime_count,
                "failed_count": failed_count,
                "used_attempts": used_attempts,
            },
        )

    def policy_vector(
        self,
        rng: np.random.Generator | None = None,
        policy: str = "distance",
        noise: float = 0.0,
        allow_window_aware: bool = False,
    ) -> np.ndarray:
        """Build a simple heuristic vector.

        This replaces the overly strong v1 greedy seed. The default policies do
        not directly exploit the continuous task-key structure to construct an
        oracle-like solution. They choose tasks using simple, transparent route
        rules and only set approximate start times.

        policy options:
        - distance: nearest unvisited task.
        - reward_density: reward divided by travel-time proxy.
        - mixed: stochastic mixture used by HSBO pre-search.
        """
        if rng is None:
            rng = np.random.default_rng(0)
        upper = np.zeros(self.dim_upper, dtype=float)
        lower = np.zeros(self.dim_lower, dtype=float)
        task_used = set()

        # Set a rough heading toward the global task centroid, not toward the exact
        # selected task sequence.
        weights = np.maximum(self.task_rewards, 1e-3)
        centroid = np.average(self.task_xy, axis=0, weights=weights)
        for i in range(self.n):
            upper[2 * i] = 0.78 if policy != "distance" else 0.70
            vec = centroid - self.uav_xy[i]
            heading = np.arctan2(vec[1], vec[0]) % (2.0 * np.pi)
            upper[2 * i + 1] = heading / (2.0 * np.pi)

        cursor = 0
        for i in range(self.n):
            pos = self.uav_xy[i].copy()
            current_time = 0.0
            for slot in range(self.k):
                candidates = []
                speed = self.v_min[i] + upper[2 * i] * (self.v_max[i] - self.v_min[i])
                for j in range(self.m):
                    if j in task_used and self.task_type[j] != "cooperative":
                        continue
                    dist = float(np.linalg.norm(self.task_xy[j] - pos))
                    eta = current_time + dist / max(speed, 1e-6)
                    # By default, the heuristic is intentionally myopic: it does
                    # not filter exactly by time-window feasibility. This avoids
                    # an oracle-like greedy baseline.
                    if policy == "distance":
                        score = -dist
                    elif policy == "reward_density":
                        type_bonus = 1.15 if self.task_type[j] in {"critical", "cooperative"} else 1.0
                        score = type_bonus * self.task_rewards[j] / (1.0 + dist + 0.05 * eta)
                    else:  # mixed
                        score = 0.55 * self.task_rewards[j] / (1.0 + dist) - 0.45 * abs(eta - 0.5 * (self.task_earliest[j] + self.task_latest[j])) / self.horizon
                    if allow_window_aware:
                        # HSBO pre-search may use a weak window-aware bias, but
                        # still not exact synchronization for cooperative tasks.
                        center = 0.5 * (self.task_earliest[j] + self.task_latest[j])
                        score -= 0.50 * abs(eta - center) / self.horizon
                    candidates.append((score, j, eta, dist))

                if not candidates:
                    j = int(rng.integers(0, self.m))
                    proposed_start = float(rng.uniform(0.0, self.horizon))
                else:
                    # Randomized tie-breaking prevents deterministic memorization.
                    candidates = sorted(candidates, key=lambda z: z[0], reverse=True)
                    rank = 0 if rng.random() < 0.80 else int(rng.integers(0, min(4, len(candidates))))
                    _, j, eta, dist = candidates[rank]
                    if self.task_type[j] != "cooperative":
                        task_used.add(j)
                    center = 0.5 * (self.task_earliest[j] + self.task_latest[j])
                    if allow_window_aware:
                        # Weak pre-search may use the center of the window, but it
                        # still does not solve cooperative synchronization exactly.
                        proposed_start = max(eta, center)
                    else:
                        proposed_start = eta
                    pos = self.task_xy[j].copy()
                    current_time = min(self.horizon, max(eta, proposed_start) + self.task_service[j])

                # Encoding is unavoidable because the simulator consumes a vector,
                # but the selected task is produced by a simple policy rather than
                # by reading a hidden task-key oracle.
                lower[cursor] = np.clip((j + 0.5) / self.m, 0.0, 1.0)
                lower[cursor + 1] = np.clip(proposed_start / self.horizon, 0.0, 1.0)
                cursor += 2

        x = np.concatenate([upper, lower])
        if noise > 0:
            # Perturb task keys moderately and timing more strongly.
            eps = rng.normal(0.0, noise, size=x.shape)
            key_indices = self.dim_upper + np.arange(0, self.dim_lower, 2)
            eps[key_indices] *= 0.35
            x = np.clip(x + eps, 0.0, 1.0)
        return x

    # Backward-compatible alias used by v1 algorithms. It now maps to a weak,
    # policy-based seed instead of the v1 oracle-like greedy vector.
    def greedy_vector(self, rng: np.random.Generator | None = None, noise: float = 0.0) -> np.ndarray:
        return self.policy_vector(rng=rng, policy="mixed", noise=noise, allow_window_aware=True)
