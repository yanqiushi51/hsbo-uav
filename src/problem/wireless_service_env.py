from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from .wireless_generator import WirelessInstance


@dataclass
class WirelessEvaluationResult:
    reward: float
    feasible: bool
    served_tasks: int
    total_travel: float
    conflict_penalty: float
    delay_penalty: float
    energy_penalty: float
    details: Dict

    @property
    def served_ratio(self) -> float:
        return float(self.details.get("served_ratio", 0.0))


class WirelessServiceEnv:
    """Wireless packet-service evaluator for event-triggered IoT collection.

    Decision vector:
    - Upper mobility: [speed_i, heading_i] for each UAV.
    - Lower service: [node_key_ik, service_start_ik] for each UAV slot.

    A packet is served only when the selected service start lies in the packet
    hard window and the UAV-node link meets both rate and upload-size gates.
    """

    def __init__(
        self,
        instance: WirelessInstance,
        channel_model: str = "distance",
        rate_threshold_mbps: float | None = None,
        lambda_e: float = 0.01,
        lambda_c: float = 2.0,
    ):
        self.instance = instance
        self.channel_model = "los_nlos_expected" if channel_model == "los_nlos" else channel_model
        self.n = instance.n_uavs
        self.m = instance.n_nodes
        self.k = instance.max_slots_per_uav
        self.dim_upper = 2 * self.n
        self.dim_lower = 2 * self.n * self.k
        self.dim = self.dim_upper + self.dim_lower
        self.horizon = instance.horizon
        self.max_start = max(0.0, instance.horizon - instance.service_duration)
        self.area_size = instance.area_size
        self.altitude = instance.altitude
        self.service_duration = instance.service_duration
        self.rate_threshold_mbps = float(rate_threshold_mbps or instance.rate_threshold_mbps)
        self.lambda_e = float(lambda_e)
        self.lambda_c = float(lambda_c)

        self.uav_xy = np.array([[u.x, u.y] for u in instance.uavs], dtype=float)
        self.v_min = np.array([u.v_min for u in instance.uavs], dtype=float)
        self.v_max = np.array([u.v_max for u in instance.uavs], dtype=float)
        self.node_xy = np.array([[n.x, n.y] for n in instance.nodes], dtype=float)
        self.packet_mbits = np.array([n.packet_mbits for n in instance.nodes], dtype=float)
        self.node_priority = np.array([n.priority for n in instance.nodes], dtype=float)
        self.node_earliest = np.array([n.earliest for n in instance.nodes], dtype=float)
        self.node_latest = np.array([n.latest for n in instance.nodes], dtype=float)

        # Compatibility aliases used by the original repair/baseline utilities.
        self.task_xy = self.node_xy
        self.task_rewards = self.node_priority
        self.task_service = np.full(self.m, self.service_duration, dtype=float)
        self.task_earliest = self.node_earliest
        self.task_latest = self.node_latest

        self.bandwidth_hz = 1.0e6
        self.tx_power_dbm = 23.0
        self.beta0_db = -30.0
        self.path_loss_exponent = 2.5
        self.noise_dbm = -104.0
        self.tx_power_mw = 10.0 ** (self.tx_power_dbm / 10.0)
        self.beta0_linear = 10.0 ** (self.beta0_db / 10.0)
        self.noise_mw = 10.0 ** (self.noise_dbm / 10.0)
        self.los_a = 9.61
        self.los_b = 0.16
        self.eta_los_db = 1.0
        self.eta_nlos_db = 20.0

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
        node_keys = lower[0::2].reshape(self.n, self.k)
        start_norm = lower[1::2].reshape(self.n, self.k)
        node_ids = np.minimum((node_keys * self.m).astype(int), self.m - 1)
        starts = start_norm * self.max_start
        return node_ids, starts

    def _uav_position(self, i: int, t: float, speed: float, heading: float) -> np.ndarray:
        return self.uav_xy[i] + speed * t * np.array([np.cos(heading), np.sin(heading)])

    def _inside_area(self, pos: np.ndarray) -> bool:
        return bool(0.0 <= pos[0] <= self.area_size and 0.0 <= pos[1] <= self.area_size)

    def _channel_gain(self, distance_3d: float, elevation_deg: float) -> float:
        distance_3d = max(float(distance_3d), 1e-6)
        h = self.beta0_linear * distance_3d ** (-self.path_loss_exponent)
        if self.channel_model == "los_nlos_expected":
            p_los = 1.0 / (
                1.0 + self.los_a * np.exp(-self.los_b * (elevation_deg - self.los_a))
            )
            expected_excess_db = p_los * self.eta_los_db + (1.0 - p_los) * self.eta_nlos_db
            h *= 10.0 ** (-expected_excess_db / 10.0)
        return float(h)

    def rate_mbps(
        self,
        i: int,
        j: int,
        t: float,
        speeds: np.ndarray,
        headings: np.ndarray,
        active_count: int = 1,
    ) -> float:
        pos = self._uav_position(i, t, speeds[i], headings[i])
        ground_dist = float(np.linalg.norm(pos - self.node_xy[j]))
        d3 = float(np.sqrt(ground_dist * ground_dist + self.altitude * self.altitude))
        elevation = float(np.degrees(np.arctan2(self.altitude, max(ground_dist, 1e-6))))
        h = self._channel_gain(d3, elevation)
        snr = self.tx_power_mw * h / max(self.noise_mw, 1e-30)
        bandwidth = self.bandwidth_hz / max(1, int(active_count)) if self.channel_model == "fdma" else self.bandwidth_hz
        return float((bandwidth * np.log2(1.0 + snr)) / 1.0e6)

    def _active_counts(self, starts: np.ndarray) -> np.ndarray:
        counts = np.ones_like(starts, dtype=int)
        if self.channel_model != "fdma":
            return counts
        flat = [(i, k, starts[i, k], starts[i, k] + self.service_duration) for i in range(self.n) for k in range(self.k)]
        for i, k, s0, e0 in flat:
            count = 0
            for _, _, s1, e1 in flat:
                if s0 < e1 and s1 < e0:
                    count += 1
            counts[i, k] = max(1, count)
        return counts

    def evaluate(self, x: np.ndarray) -> WirelessEvaluationResult:
        upper, lower = self.split(x)
        speeds, headings = self.decode_upper(upper)
        node_ids, starts = self.decode_lower(lower)
        active_counts = self._active_counts(starts)

        attempts_by_node: Dict[int, List[Dict]] = {j: [] for j in range(self.m)}
        all_attempts = []
        window_count = 0
        rate_count = 0
        tx_count = 0
        outage_count = 0
        boundary_invalid_count = 0

        for i in range(self.n):
            for slot in range(self.k):
                j = int(node_ids[i, slot])
                start = float(starts[i, slot])
                pos = self._uav_position(i, start, speeds[i], headings[i])
                inside = self._inside_area(pos)
                window_ok = bool(self.node_earliest[j] <= start <= self.node_latest[j])
                rate = self.rate_mbps(i, j, start, speeds, headings, int(active_counts[i, slot]))
                rate_ok = bool(inside and rate >= self.rate_threshold_mbps)
                tx_ok = bool(rate_ok and self.service_duration * rate >= self.packet_mbits[j])
                success = bool(window_ok and tx_ok)

                window_count += int(window_ok)
                rate_count += int(window_ok and rate_ok)
                tx_count += int(success)
                outage_count += int(window_ok and inside and not tx_ok)
                boundary_invalid_count += int(not inside)

                attempt = {
                    "uav": i,
                    "slot": slot,
                    "node": j,
                    "start": start,
                    "finish": start + self.service_duration,
                    "rate_mbps": rate,
                    "active_count": int(active_counts[i, slot]),
                    "inside_area": inside,
                    "window_ok": window_ok,
                    "rate_ok": rate_ok,
                    "transmission_ok": tx_ok,
                    "success": success,
                }
                attempts_by_node[j].append(attempt)
                all_attempts.append(attempt)

        conflict_count = 0
        for i in range(self.n):
            own = sorted([a for a in all_attempts if a["uav"] == i], key=lambda z: z["start"])
            for a, b in zip(own, own[1:]):
                if b["start"] - a["start"] < self.service_duration:
                    conflict_count += 1

        served_nodes = []
        duplicate_count = 0
        for j, attempts in attempts_by_node.items():
            successful = [a for a in attempts if a["success"]]
            if successful:
                served_nodes.append(j)
                duplicate_count += max(0, len(successful) - 1)

        raw_utility = float(np.sum(self.node_priority[served_nodes])) if served_nodes else 0.0
        total_travel = float(np.sum(speeds) * self.horizon)
        energy_cost = total_travel / 100.0
        conflict_cost = duplicate_count + conflict_count
        utility = max(0.0, raw_utility - self.lambda_e * energy_cost - self.lambda_c * conflict_cost)
        feasible = bool(utility > 0.0 and len(served_nodes) > 0)

        details = {
            "served_global": sorted(int(j) for j in served_nodes),
            "served_ratio": len(served_nodes) / max(1, self.m),
            "attempts": all_attempts,
            "window_feasible_attempts": window_count,
            "rate_feasible_attempts": rate_count,
            "transmission_feasible_attempts": tx_count,
            "outage_count": outage_count,
            "duplicate_count": duplicate_count,
            "conflict_count": conflict_count,
            "boundary_invalid_count": boundary_invalid_count,
            "energy_cost": energy_cost,
            "raw_packet_utility": raw_utility,
            "channel_model": self.channel_model,
            "window_width": self.instance.window_width,
            "rate_threshold_mbps": self.rate_threshold_mbps,
        }
        return WirelessEvaluationResult(
            reward=float(utility),
            feasible=feasible,
            served_tasks=len(served_nodes),
            total_travel=total_travel,
            conflict_penalty=float(self.lambda_c * conflict_cost),
            delay_penalty=0.0,
            energy_penalty=float(self.lambda_e * energy_cost),
            details=details,
        )

    def _best_time_for_node(
        self,
        i: int,
        j: int,
        speeds: np.ndarray,
        headings: np.ndarray,
        lower_bound: float = 0.0,
    ) -> tuple[float, float, bool]:
        lo = max(float(self.node_earliest[j]), float(lower_bound), 0.0)
        hi = min(float(self.node_latest[j]), self.max_start)
        if hi < lo:
            t = min(max(lo, 0.0), self.max_start)
            return t, self.rate_mbps(i, j, t, speeds, headings), False
        grid = np.linspace(lo, hi, num=5)
        rates = np.array([self.rate_mbps(i, j, float(t), speeds, headings) for t in grid])
        idx = int(np.argmax(rates))
        t = float(grid[idx])
        pos = self._uav_position(i, t, speeds[i], headings[i])
        feasible = self._inside_area(pos) and rates[idx] >= self.rate_threshold_mbps and self.service_duration * rates[idx] >= self.packet_mbits[j]
        return t, float(rates[idx]), bool(feasible)

    def _encode_lower_entry(self, lower: np.ndarray, idx: int, j: int, start: float) -> None:
        lower[idx] = np.clip((j + 0.5) / self.m, 0.0, 1.0)
        lower[idx + 1] = np.clip(start / max(self.max_start, 1e-9), 0.0, 1.0)

    def policy_vector(
        self,
        rng: np.random.Generator | None = None,
        policy: str = "mixed",
        noise: float = 0.0,
        allow_window_aware: bool = False,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(0)
        policy = policy.lower()
        upper = np.zeros(self.dim_upper, dtype=float)
        lower = np.zeros(self.dim_lower, dtype=float)

        weights = np.maximum(self.node_priority, 1e-3)
        centroid = np.average(self.node_xy, axis=0, weights=weights)
        assigned_targets = set()
        for i in range(self.n):
            candidates = []
            for j in range(self.m):
                center = 0.5 * (self.node_earliest[j] + self.node_latest[j])
                dist = float(np.linalg.norm(self.node_xy[j] - self.uav_xy[i]))
                required_speed = dist / max(center, 1.0)
                speed_gap = abs(required_speed - 0.5 * (self.v_min[i] + self.v_max[i]))
                speed_ok = self.v_min[i] <= required_speed <= self.v_max[i]
                if policy == "edf_rate":
                    score = -self.node_latest[j] / self.horizon + 2.0 * int(speed_ok)
                elif policy in {"nearest_feasible", "distance"}:
                    score = -dist / self.area_size + 2.0 * int(speed_ok)
                elif policy == "max_rate":
                    score = -speed_gap / max(self.v_max[i], 1.0) + 0.05 * self.node_priority[j] + 2.0 * int(speed_ok)
                else:
                    score = 0.15 * self.node_priority[j] - speed_gap / max(self.v_max[i], 1.0) + 2.0 * int(speed_ok)
                if j in assigned_targets:
                    score -= 0.75
                candidates.append((score, j, required_speed))
            candidates.sort(key=lambda z: z[0], reverse=True)
            _, target_j, required_speed = candidates[0]
            assigned_targets.add(int(target_j))
            target = self.node_xy[target_j]
            speed = float(np.clip(required_speed, self.v_min[i], self.v_max[i]))
            upper[2 * i] = np.clip((speed - self.v_min[i]) / max(self.v_max[i] - self.v_min[i], 1e-9), 0.0, 1.0)
            if not np.isfinite(upper[2 * i]):
                upper[2 * i] = 0.70
            heading = np.arctan2(target[1] - self.uav_xy[i, 1], target[0] - self.uav_xy[i, 0]) % (2.0 * np.pi)
            if not np.isfinite(heading):
                heading = np.arctan2(centroid[1] - self.uav_xy[i, 1], centroid[0] - self.uav_xy[i, 0]) % (2.0 * np.pi)
            upper[2 * i + 1] = heading / (2.0 * np.pi)

        speeds, headings = self.decode_upper(upper)
        used_nodes = set()
        cursor = 0
        for i in range(self.n):
            min_start = 0.0
            for _slot in range(self.k):
                candidates = []
                for j in range(self.m):
                    if j in used_nodes:
                        continue
                    if allow_window_aware or policy in {"max_rate", "edf_rate", "nearest_feasible", "reward_density"}:
                        t, rate, feasible = self._best_time_for_node(i, j, speeds, headings, lower_bound=min_start)
                    else:
                        t = float(np.clip((len(used_nodes) + 0.5) / max(1, self.m) * self.max_start, 0.0, self.max_start))
                        rate = self.rate_mbps(i, j, t, speeds, headings)
                        feasible = False
                    dist = float(np.linalg.norm(self._uav_position(i, t, speeds[i], headings[i]) - self.node_xy[j]))
                    if policy == "max_rate":
                        score = rate + 0.05 * self.node_priority[j] + (5.0 if feasible else 0.0)
                    elif policy == "edf_rate":
                        score = -self.node_latest[j] / self.horizon + (5.0 if feasible else 0.0) + 0.02 * rate
                    elif policy in {"nearest_feasible", "distance"}:
                        score = -dist / self.area_size + (5.0 if feasible else 0.0)
                    elif policy == "reward_density":
                        score = self.node_priority[j] * max(rate, 1e-6) / (1.0 + dist / 100.0)
                    else:
                        score = 0.55 * self.node_priority[j] + 0.35 * rate - 0.10 * dist / self.area_size
                        if feasible:
                            score += 3.0
                    candidates.append((score, j, t, feasible))
                if candidates:
                    candidates.sort(key=lambda z: z[0], reverse=True)
                    rank = 0 if rng.random() < 0.85 else int(rng.integers(0, min(4, len(candidates))))
                    _, j, start, _ = candidates[rank]
                    used_nodes.add(j)
                else:
                    j = int(rng.integers(0, self.m))
                    start = float(rng.uniform(0.0, self.max_start))
                self._encode_lower_entry(lower, cursor, int(j), float(start))
                min_start = float(start) + self.service_duration
                cursor += 2

        x = np.concatenate([upper, lower])
        if noise > 0.0:
            eps = rng.normal(0.0, noise, size=x.shape)
            key_indices = self.dim_upper + np.arange(0, self.dim_lower, 2)
            eps[key_indices] *= 0.35
            x = np.clip(x + eps, 0.0, 1.0)
        return x

    def greedy_vector(self, rng: np.random.Generator | None = None, noise: float = 0.0) -> np.ndarray:
        return self.policy_vector(rng=rng, policy="max_rate", noise=noise, allow_window_aware=True)

    def repair_vector(self, x: np.ndarray, strength: float = 0.85) -> np.ndarray:
        upper, lower = self.split(x)
        repaired = lower.copy()
        speeds, headings = self.decode_upper(upper)
        node_ids, starts = self.decode_lower(lower)
        cursor = 0
        for i in range(self.n):
            min_start = 0.0
            for slot in range(self.k):
                j = int(node_ids[i, slot])
                best_t, _rate, feasible = self._best_time_for_node(i, j, speeds, headings, lower_bound=min_start)
                if not feasible:
                    best = None
                    for cand in range(self.m):
                        t, rate, ok = self._best_time_for_node(i, cand, speeds, headings, lower_bound=min_start)
                        score = (10.0 if ok else 0.0) + self.node_priority[cand] + 0.15 * rate
                        if best is None or score > best[0]:
                            best = (score, cand, t)
                    if best is not None:
                        j = int(best[1])
                        best_t = float(best[2])
                old_t = float(starts[i, slot])
                start = strength * best_t + (1.0 - strength) * old_t
                self._encode_lower_entry(repaired, cursor, j, start)
                min_start = start + self.service_duration
                cursor += 2
        return np.concatenate([upper, repaired])
