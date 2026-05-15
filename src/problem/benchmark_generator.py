from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List
import json
import numpy as np


@dataclass
class UAV:
    id: int
    x: float
    y: float
    v_min: float
    v_max: float


@dataclass
class Task:
    id: int
    x: float
    y: float
    reward: float
    service_time: float
    earliest: float
    latest: float
    risk: float
    task_type: str = "ordinary"  # ordinary | critical | cooperative
    required_uavs: int = 1
    sync_tolerance: float = 0.0


@dataclass
class BenchmarkInstance:
    scale: str
    seed: int
    n_uavs: int
    n_tasks: int
    max_actions_per_uav: int
    horizon: float
    uavs: List[UAV]
    tasks: List[Task]

    @property
    def dimension(self) -> int:
        return 2 * self.n_uavs + 2 * self.n_uavs * self.max_actions_per_uav

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["dimension"] = self.dimension
        return data


def _scale_config(scale: str) -> Dict[str, float]:
    # v2 deliberately tightens time windows and increases cooperative tasks so
    # that the benchmark behaves like a sparse-reward scheduling problem rather
    # than an easy routing problem.
    configs = {
        "small": dict(
            n_uavs=3,
            n_tasks=10,
            max_actions_per_uav=2,
            horizon=120.0,
            area=1000.0,
            width_range=(0.040, 0.080),
            critical_ratio=0.30,
            cooperative_ratio=0.20,
            coop_sync=6.0,
        ),
        "medium": dict(
            n_uavs=5,
            n_tasks=20,
            max_actions_per_uav=3,
            horizon=180.0,
            area=1600.0,
            width_range=(0.025, 0.055),
            critical_ratio=0.35,
            cooperative_ratio=0.25,
            coop_sync=5.0,
        ),
        "large": dict(
            n_uavs=8,
            n_tasks=40,
            max_actions_per_uav=4,
            horizon=260.0,
            area=2400.0,
            width_range=(0.018, 0.045),
            critical_ratio=0.40,
            cooperative_ratio=0.30,
            coop_sync=4.0,
        ),
    }
    if scale not in configs:
        raise ValueError(f"Unknown scale {scale!r}; choose from {list(configs)}")
    return configs[scale]


def generate_instance(scale: str, seed: int = 0) -> BenchmarkInstance:
    """Generate a reproducible sparse-reward multi-UAV scheduling instance.

    Version 2 adds two mechanisms that prevent a simple greedy route from
    dominating the benchmark:

    1) Narrow time windows: a task contributes reward only if the proposed
       service time falls inside a tight window.
    2) Cooperative tasks: selected high-value tasks require multiple UAVs to
       arrive within a synchronization tolerance. A single UAV receives no
       reward for such a task.

    These mechanisms mimic the sparse, coupled reward structure of real
    multi-agent mission scheduling while remaining safe and reproducible.
    """
    cfg = _scale_config(scale)
    rng = np.random.default_rng(seed)
    area = cfg["area"]
    horizon = cfg["horizon"]
    n_uavs = int(cfg["n_uavs"])
    n_tasks = int(cfg["n_tasks"])
    k = int(cfg["max_actions_per_uav"])

    uavs: List[UAV] = []
    for i in range(n_uavs):
        x = rng.uniform(0.03 * area, 0.18 * area)
        y = rng.uniform(0.05 * area, 0.95 * area)
        uavs.append(UAV(id=i, x=float(x), y=float(y), v_min=8.0, v_max=28.0))

    tasks: List[Task] = []
    n_coop = max(1, int(round(n_tasks * cfg["cooperative_ratio"])))
    n_critical = max(1, int(round(n_tasks * cfg["critical_ratio"])))
    task_order = rng.permutation(n_tasks)
    coop_ids = set(int(x) for x in task_order[:n_coop])
    critical_ids = set(int(x) for x in task_order[n_coop : n_coop + n_critical])

    width_low, width_high = cfg["width_range"]
    for j in range(n_tasks):
        x = rng.uniform(0.22 * area, 0.98 * area)
        y = rng.uniform(0.04 * area, 0.96 * area)

        center = rng.uniform(0.18 * horizon, 0.82 * horizon)
        width = rng.uniform(width_low * horizon, width_high * horizon)
        # Critical/cooperative tasks get narrower windows and higher rewards.
        task_type = "ordinary"
        required_uavs = 1
        sync_tolerance = 0.0
        reward_mult = 1.0
        if j in coop_ids:
            task_type = "cooperative"
            required_uavs = 2 if n_uavs < 6 else int(rng.choice([2, 3], p=[0.75, 0.25]))
            width *= 0.75
            sync_tolerance = float(cfg["coop_sync"])
            reward_mult = 2.2 + 0.25 * required_uavs
        elif j in critical_ids:
            task_type = "critical"
            width *= 0.65
            reward_mult = 1.5

        earliest = max(0.0, center - width / 2.0 + rng.normal(0.0, 0.015 * horizon))
        latest = min(horizon, center + width / 2.0 + rng.normal(0.0, 0.015 * horizon))
        min_width = (0.012 if scale == "large" else 0.018) * horizon
        if latest - earliest < min_width:
            latest = min(horizon, earliest + min_width)

        reward = rng.uniform(8.0, 35.0) * (1.0 + 0.3 * rng.random()) * reward_mult
        service_time = rng.uniform(4.0, 13.0)
        if task_type in {"critical", "cooperative"}:
            service_time *= rng.uniform(0.75, 1.15)
        risk = rng.uniform(0.0, 1.0)
        tasks.append(
            Task(
                id=j,
                x=float(x),
                y=float(y),
                reward=float(reward),
                service_time=float(service_time),
                earliest=float(earliest),
                latest=float(latest),
                risk=float(risk),
                task_type=task_type,
                required_uavs=required_uavs,
                sync_tolerance=sync_tolerance,
            )
        )

    return BenchmarkInstance(
        scale=scale,
        seed=seed,
        n_uavs=n_uavs,
        n_tasks=n_tasks,
        max_actions_per_uav=k,
        horizon=float(horizon),
        uavs=uavs,
        tasks=tasks,
    )


def save_instance(instance: BenchmarkInstance, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(instance.to_dict(), indent=2), encoding="utf-8")


def load_instance(path: str | Path) -> BenchmarkInstance:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    uavs = [UAV(**u) for u in data["uavs"]]
    tasks = [Task(**t) for t in data["tasks"]]
    return BenchmarkInstance(
        scale=data["scale"],
        seed=int(data["seed"]),
        n_uavs=int(data["n_uavs"]),
        n_tasks=int(data["n_tasks"]),
        max_actions_per_uav=int(data["max_actions_per_uav"]),
        horizon=float(data["horizon"]),
        uavs=uavs,
        tasks=tasks,
    )


def generate_all(output_dir: str | Path, seeds=(0, 1, 2)) -> None:
    output_dir = Path(output_dir)
    for scale in ("small", "medium", "large"):
        for seed in seeds:
            instance = generate_instance(scale, seed)
            save_instance(instance, output_dir / scale / f"instance_seed{seed}.json")
