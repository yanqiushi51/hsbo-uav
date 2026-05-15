from __future__ import annotations

from pathlib import Path
import argparse
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.problem.wireless_generator import (
    budget_for_scale,
    generate_wireless_instance,
    load_wireless_instance,
    save_wireless_instance,
)
from src.problem.wireless_service_env import WirelessServiceEnv
from src.baselines import (
    run_random_search,
    run_ga,
    run_pso,
    run_standard_bo,
    run_cmaes,
    run_cmaes_repair,
    run_de,
    run_de_repair,
    run_de_repair_seeded,
    run_max_rate_greedy,
    run_edf_rate_greedy,
    run_nearest_feasible_user,
    run_rate_window_repair_bo,
)
from src.hsbo.hsbo_full import run_hsbo
from src.baselines.surrogate_bo_core import run_full_space_refinement_bo


def _run_ft_hsbo(env, budget: int, seed: int, method_name: str):
    return run_hsbo(env, budget=budget, seed=seed, method_name=method_name)


def _run_ft_hsbo_no_hierarchy(env, budget: int, seed: int, method_name: str):
    return run_hsbo(env, budget=budget, seed=seed, method_name=method_name, use_hierarchy=False)


def _run_ft_hsbo_no_presearch(env, budget: int, seed: int, method_name: str):
    return run_hsbo(env, budget=budget, seed=seed, method_name=method_name, use_presearch=False)


def _run_ft_hsbo_no_hierarchy_clean(env, budget: int, seed: int, method_name: str):
    return run_full_space_refinement_bo(env, budget=budget, seed=seed, method_name=method_name)


METHODS = {
    "ft_hsbo": _run_ft_hsbo,
    "ft_hsbo_no_hierarchy": _run_ft_hsbo_no_hierarchy,
    "ft_hsbo_no_hierarchy_clean": _run_ft_hsbo_no_hierarchy_clean,
    "ft_hsbo_no_presearch": _run_ft_hsbo_no_presearch,
    "full_space_bo": run_standard_bo,
    "rs": run_random_search,
    "ga": run_ga,
    "pso": run_pso,
    "cmaes": run_cmaes,
    "cmaes_r": run_cmaes_repair,
    "de": run_de,
    "de_r": run_de_repair,
    "de_rp": run_de_repair_seeded,
    "max_rate_greedy": run_max_rate_greedy,
    "edf_rate_greedy": run_edf_rate_greedy,
    "nearest_feasible_user": run_nearest_feasible_user,
    "repair_bo": run_rate_window_repair_bo,
}

MAIN_METHODS = [
    "ft_hsbo",
    "full_space_bo",
    "rs",
    "ga",
    "pso",
    "cmaes",
    "de",
    "cmaes_r",
    "de_r",
    "de_rp",
    "max_rate_greedy",
    "edf_rate_greedy",
    "nearest_feasible_user",
    "repair_bo",
]
ABLATION_METHODS = [
    "ft_hsbo",
    "ft_hsbo_no_hierarchy_clean",
    "ft_hsbo_no_presearch",
    "full_space_bo",
    "repair_bo",
]
SENSITIVITY_METHODS = [
    "ft_hsbo",
    "full_space_bo",
    "de_rp",
    "edf_rate_greedy",
    "max_rate_greedy",
]
BUDGET_METHODS = ["ft_hsbo", "full_space_bo", "de_rp", "edf_rate_greedy", "pso"]


def setting_dir(window_width: float, rate_threshold: float) -> str:
    w = str(window_width).replace(".", "p")
    r = str(rate_threshold).replace(".", "p")
    return f"w{w}_r{r}"


def get_instance(scale: str, seed: int, data_root: Path, window_width: float, rate_threshold: float):
    path = data_root / setting_dir(window_width, rate_threshold) / scale / f"instance_seed{seed}.json"
    if path.exists():
        return load_wireless_instance(path)
    instance = generate_wireless_instance(
        scale=scale,
        seed=seed,
        window_width=window_width,
        rate_threshold_mbps=rate_threshold,
    )
    save_wireless_instance(instance, path)
    return instance


def run_method(
    experiment: str,
    scale: str,
    method: str,
    seed: int,
    budget: int,
    data_root: Path,
    logs_root: Path,
    window_width: float,
    rate_threshold: float,
    channel_model: str,
    log_experiment: str | None = None,
) -> Path:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {list(METHODS)}")
    out = logs_root / (log_experiment or experiment) / scale / method / f"seed_{seed}.csv"
    if out.exists():
        try:
            existing = pd.read_csv(out)
            if len(existing) >= int(budget):
                print(f"[skip] {out}")
                return out
        except Exception:
            pass

    instance = get_instance(scale, seed, data_root, window_width, rate_threshold)
    env = WirelessServiceEnv(instance, channel_model=channel_model, rate_threshold_mbps=rate_threshold)
    df = METHODS[method](env, budget=budget, seed=seed, method_name=method)
    df["experiment"] = experiment
    df["channel_model"] = channel_model
    df["budget"] = int(budget)
    df["window_width"] = float(window_width)
    df["rate_threshold_mbps"] = float(rate_threshold)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def summarize_audit(paths: list[Path], out_path: Path, samples: int) -> None:
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        scale = str(df["scale"].iloc[0])
        rows.append(
            {
                "scale": scale,
                "random_samples": int(len(df)),
                "window_feasible_pct": 100.0 * (df["window_feasible_attempts"] > 0).mean(),
                "rate_feasible_pct": 100.0 * (df["rate_feasible_attempts"] > 0).mean(),
                "transmission_feasible_pct": 100.0 * (df["transmission_feasible_attempts"] > 0).mean(),
                "positive_utility_pct": 100.0 * df["success_flag"].astype(bool).mean(),
                "target_samples": int(samples),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def run_grid(
    experiment: str,
    scales: list[str],
    methods: list[str],
    seeds_by_scale: dict[str, list[int]],
    budget_by_scale: dict[str, int],
    args,
    window_width: float,
    rate_threshold: float,
    channel_model: str,
    log_experiment: str | None = None,
) -> list[Path]:
    paths = []
    for scale in scales:
        for method in methods:
            for seed in seeds_by_scale[scale]:
                print(f"[{experiment}] scale={scale} method={method} seed={seed} budget={budget_by_scale[scale]}")
                path = run_method(
                    experiment=experiment,
                    scale=scale,
                    method=method,
                    seed=seed,
                    budget=budget_by_scale[scale],
                    data_root=ROOT / args.data_root,
                    logs_root=ROOT / args.logs_root,
                    window_width=window_width,
                    rate_threshold=rate_threshold,
                    channel_model=channel_model,
                    log_experiment=log_experiment,
                )
                paths.append(path)
    return paths


def default_seeds(scale: str) -> list[int]:
    return list(range(5)) if scale == "large" else list(range(10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["pilot", "sparsity", "main", "ablation", "window", "rate", "channel", "budget", "minimal"],
        required=True,
    )
    parser.add_argument("--data-root", default="archive/data/datasets_wireless")
    parser.add_argument("--logs-root", default="outputs/logs")
    parser.add_argument("--results-root", default="outputs/results")
    parser.add_argument("--scales", "--scale", nargs="+", choices=["small", "medium", "large"], dest="scales")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--window-width", "--window_width", type=float, default=30.0, dest="window_width")
    parser.add_argument("--rate-threshold-mbps", "--rate_threshold_mbps", type=float, default=1.0, dest="rate_threshold_mbps")
    parser.add_argument("--rate-values", nargs="+", type=float)
    parser.add_argument(
        "--channel-model",
        "--channel_model",
        choices=["distance", "los_nlos", "los_nlos_expected", "fdma"],
        default="distance",
        dest="channel_model",
    )
    args = parser.parse_args()

    results_root = ROOT / args.results_root
    paths: list[Path] = []

    if args.experiment == "pilot":
        seeds = args.seeds or [0]
        budget = args.budget or args.samples
        paths = run_grid(
            "pilot",
            ["medium"],
            ["rs"],
            {"medium": seeds},
            {"medium": budget},
            args,
            args.window_width,
            args.rate_threshold_mbps,
            args.channel_model,
        )
        summarize_audit(paths, results_root / "pilot_calibration.csv", budget)
        return

    if args.experiment == "sparsity":
        scales = args.scales or ["small", "medium", "large"]
        seeds_by_scale = {s: args.seeds or [0] for s in scales}
        budget_by_scale = {s: args.budget or args.samples for s in scales}
        paths = run_grid(
            "sparsity",
            scales,
            ["rs"],
            seeds_by_scale,
            budget_by_scale,
            args,
            args.window_width,
            args.rate_threshold_mbps,
            args.channel_model,
        )
        summarize_audit(paths, results_root / "summary_sparsity_audit.csv", args.samples)
        return

    if args.experiment == "main":
        scales = args.scales or ["medium", "large", "small"]
        seeds_by_scale = {s: args.seeds or default_seeds(s) for s in scales}
        budget_by_scale = {s: args.budget or budget_for_scale(s) for s in scales}
        methods = args.methods or MAIN_METHODS
        run_grid("main", scales, methods, seeds_by_scale, budget_by_scale, args, args.window_width, args.rate_threshold_mbps, "distance")
        return

    if args.experiment == "ablation":
        seeds = args.seeds or list(range(10))
        methods = args.methods or ABLATION_METHODS
        run_grid("ablation", ["medium"], methods, {"medium": seeds}, {"medium": args.budget or 300}, args, args.window_width, args.rate_threshold_mbps, "distance")
        return

    if args.experiment == "window":
        seeds = args.seeds or list(range(10))
        methods = args.methods or SENSITIVITY_METHODS
        for label, width in [("wide", 60.0), ("medium", 30.0), ("narrow", 15.0)]:
            run_grid(f"window_{label}", ["medium"], methods, {"medium": seeds}, {"medium": args.budget or 300}, args, width, args.rate_threshold_mbps, "distance")
        return

    if args.experiment == "rate":
        seeds = args.seeds or list(range(10))
        methods = args.methods or SENSITIVITY_METHODS
        rate_values = args.rate_values or [0.5, 1.0, 2.0]
        labels = ["low", "medium", "high"] if len(rate_values) == 3 else [str(v).replace(".", "p") for v in rate_values]
        for label, rate in zip(labels, rate_values):
            run_grid(f"rate_{label}", ["medium"], methods, {"medium": seeds}, {"medium": args.budget or 300}, args, 30.0, rate, "distance")
        return

    if args.experiment == "channel":
        seeds = args.seeds or list(range(10))
        methods = args.methods or SENSITIVITY_METHODS
        scales = args.scales or ["medium"]
        channel = "los_nlos_expected" if args.channel_model == "los_nlos" else args.channel_model
        experiment = "channel_los_nlos" if channel == "los_nlos_expected" else f"channel_{channel}"
        run_grid(
            experiment,
            scales,
            methods,
            {s: seeds for s in scales},
            {s: args.budget or 300 for s in scales},
            args,
            args.window_width,
            args.rate_threshold_mbps,
            channel,
            log_experiment="channel",
        )
        return

    if args.experiment == "budget":
        seeds = args.seeds or list(range(10))
        methods = args.methods or BUDGET_METHODS
        for budget in [100, 200, 300, 500, 800]:
            run_grid(f"budget_{budget}", ["medium"], methods, {"medium": seeds}, {"medium": budget}, args, args.window_width, args.rate_threshold_mbps, "distance")
        return

    if args.experiment == "minimal":
        # The minimum package requested for a WCL-ready experimental spine.
        main_args = args
        run_grid("sparsity", ["small", "medium", "large"], ["rs"], {s: [0] for s in ["small", "medium", "large"]}, {s: args.samples for s in ["small", "medium", "large"]}, main_args, args.window_width, args.rate_threshold_mbps, "distance")
        run_grid("main", ["medium", "large"], args.methods or MAIN_METHODS, {"medium": list(range(10)), "large": list(range(5))}, {"medium": 300, "large": 300}, main_args, args.window_width, args.rate_threshold_mbps, "distance")
        run_grid("ablation", ["medium"], ABLATION_METHODS, {"medium": list(range(10))}, {"medium": 300}, main_args, args.window_width, args.rate_threshold_mbps, "distance")
        for label, width in [("wide", 60.0), ("medium", 30.0), ("narrow", 15.0)]:
            run_grid(f"window_{label}", ["medium"], SENSITIVITY_METHODS, {"medium": list(range(10))}, {"medium": 300}, main_args, width, args.rate_threshold_mbps, "distance")
        rate_values = args.rate_values or [0.5, 1.0, 2.0]
        labels = ["low", "medium", "high"] if len(rate_values) == 3 else [str(v).replace(".", "p") for v in rate_values]
        for label, rate in zip(labels, rate_values):
            run_grid(f"rate_{label}", ["medium"], SENSITIVITY_METHODS, {"medium": list(range(10))}, {"medium": 300}, main_args, 30.0, rate, "distance")


if __name__ == "__main__":
    main()
