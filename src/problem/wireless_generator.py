from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List
import json
import numpy as np


@dataclass
class WirelessUAV:
    id: int
    x: float
    y: float
    v_min: float
    v_max: float


@dataclass
class IoTNode:
    id: int
    x: float
    y: float
    packet_mbits: float
    priority: float
    earliest: float
    latest: float


@dataclass
class WirelessInstance:
    scale: str
    seed: int
    n_uavs: int
    n_nodes: int
    max_slots_per_uav: int
    horizon: float
    area_size: float
    altitude: float
    service_duration: float
    window_width: float
    rate_threshold_mbps: float
    uavs: List[WirelessUAV]
    nodes: List[IoTNode]

    @property
    def n_tasks(self) -> int:
        return self.n_nodes

    @property
    def max_actions_per_uav(self) -> int:
        return self.max_slots_per_uav

    @property
    def dimension(self) -> int:
        return 2 * self.n_uavs + 2 * self.n_uavs * self.max_slots_per_uav

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["dimension"] = self.dimension
        return data


def scale_config(scale: str) -> Dict[str, int]:
    configs = {
        "small": dict(n_uavs=3, n_nodes=10, max_slots_per_uav=2, budget=150),
        "medium": dict(n_uavs=5, n_nodes=20, max_slots_per_uav=3, budget=300),
        "large": dict(n_uavs=8, n_nodes=40, max_slots_per_uav=4, budget=300),
    }
    if scale not in configs:
        raise ValueError(f"Unknown scale {scale!r}; choose from {list(configs)}")
    return configs[scale]


def budget_for_scale(scale: str) -> int:
    return int(scale_config(scale)["budget"])


def _boundary_point(area: float, u: float) -> tuple[float, float]:
    """Map u in [0, 1) to an evenly spaced point on the square boundary."""
    p = (u % 1.0) * 4.0 * area
    if p < area:
        return p, 0.0
    if p < 2.0 * area:
        return area, p - area
    if p < 3.0 * area:
        return area - (p - 2.0 * area), area
    return 0.0, area - (p - 3.0 * area)


def generate_wireless_instance(
    scale: str,
    seed: int = 0,
    window_width: float = 30.0,
    rate_threshold_mbps: float = 1.0,
    horizon: float = 300.0,
    area_size: float = 1000.0,
    altitude: float = 100.0,
    service_duration: float = 5.0,
) -> WirelessInstance:
    cfg = scale_config(scale)
    rng = np.random.default_rng(seed)
    n_uavs = int(cfg["n_uavs"])
    n_nodes = int(cfg["n_nodes"])
    k = int(cfg["max_slots_per_uav"])

    uavs: List[WirelessUAV] = []
    phase = rng.uniform(0.0, 1.0)
    for i in range(n_uavs):
        x, y = _boundary_point(area_size, phase + i / max(1, n_uavs))
        jitter = rng.normal(0.0, 0.012 * area_size, size=2)
        x = float(np.clip(x + jitter[0], 0.0, area_size))
        y = float(np.clip(y + jitter[1], 0.0, area_size))
        uavs.append(WirelessUAV(id=i, x=x, y=y, v_min=5.0, v_max=25.0))

    nodes: List[IoTNode] = []
    latest_start = max(0.0, horizon - window_width)
    for j in range(n_nodes):
        a = float(rng.uniform(0.0, latest_start))
        nodes.append(
            IoTNode(
                id=j,
                x=float(rng.uniform(0.0, area_size)),
                y=float(rng.uniform(0.0, area_size)),
                packet_mbits=float(rng.uniform(1.0, 5.0)),
                priority=float(rng.uniform(5.0, 15.0)),
                earliest=a,
                latest=float(min(horizon, a + window_width)),
            )
        )

    return WirelessInstance(
        scale=scale,
        seed=int(seed),
        n_uavs=n_uavs,
        n_nodes=n_nodes,
        max_slots_per_uav=k,
        horizon=float(horizon),
        area_size=float(area_size),
        altitude=float(altitude),
        service_duration=float(service_duration),
        window_width=float(window_width),
        rate_threshold_mbps=float(rate_threshold_mbps),
        uavs=uavs,
        nodes=nodes,
    )


def save_wireless_instance(instance: WirelessInstance, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(instance.to_dict(), indent=2), encoding="utf-8")


def load_wireless_instance(path: str | Path) -> WirelessInstance:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    uavs = [WirelessUAV(**u) for u in data["uavs"]]
    nodes = [IoTNode(**n) for n in data["nodes"]]
    return WirelessInstance(
        scale=data["scale"],
        seed=int(data["seed"]),
        n_uavs=int(data["n_uavs"]),
        n_nodes=int(data["n_nodes"]),
        max_slots_per_uav=int(data["max_slots_per_uav"]),
        horizon=float(data["horizon"]),
        area_size=float(data["area_size"]),
        altitude=float(data["altitude"]),
        service_duration=float(data["service_duration"]),
        window_width=float(data["window_width"]),
        rate_threshold_mbps=float(data["rate_threshold_mbps"]),
        uavs=uavs,
        nodes=nodes,
    )


def generate_all_wireless(
    output_dir: str | Path,
    seeds=range(10),
    window_width: float = 30.0,
    rate_threshold_mbps: float = 1.0,
) -> None:
    output_dir = Path(output_dir)
    for scale in ("small", "medium", "large"):
        for seed in seeds:
            instance = generate_wireless_instance(
                scale=scale,
                seed=int(seed),
                window_width=window_width,
                rate_threshold_mbps=rate_threshold_mbps,
            )
            save_wireless_instance(instance, output_dir / scale / f"instance_seed{seed}.json")
