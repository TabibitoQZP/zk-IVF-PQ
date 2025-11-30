import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np

from zk_IVF_PQ.zk_IVF_PQ import py_fri, py_merkle
from bench.bench_suite import BenchConfig, DEFAULT_CONFIGS, RESULT_DIR


ScenarioName = Literal["fri_cluster", "fri_full", "merkle"]
MetricName = Literal["duration", "memory_peak"]


METRIC_NAMES: Tuple[MetricName, ...] = ("duration", "memory_peak")


def log2(n: int) -> int:
    cnt = 0
    while n > 1:
        n //= 2
        cnt += 1
    return cnt


def _result_file_name(scenario: ScenarioName, config: BenchConfig) -> Path:
    name = (
        f"commit_{scenario}"
        f"_N{config.N}_D{config.D}_M{config.M}_K{config.K}"
        f"_nlist{config.n_list}_nprobe{config.n_probe}"
        f"_{config.name}.json"
    )
    return RESULT_DIR / name


def _run_once(scenario: ScenarioName, config: BenchConfig) -> Dict[MetricName, float]:
    if config.N % config.n_list != 0:
        raise ValueError(f"N must be divisible by n_list for config {config.name}")
    if config.D % config.M != 0:
        raise ValueError(f"D must be divisible by M for config {config.name}")

    n_list, n, M = config.n_list, config.n, config.M

    if scenario == "fri_cluster":
        duration, memory_peak = py_fri(log2(n * M), True)
    elif scenario == "fri_full":
        duration, memory_peak = py_fri(log2(config.N * config.M), True)
    else:  # "merkle"
        duration, memory_peak = py_merkle(n_list, n, M, 64)

    result: Dict[MetricName, float] = {
        "duration": float(duration),
        "memory_peak": float(memory_peak),
    }
    return result


def _load_cached(path: Path) -> List[Dict[MetricName, float]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    runs = payload.get("runs", [])
    return [
        {metric: float(run[metric]) for metric in METRIC_NAMES}
        for run in runs
        if all(metric in run for metric in METRIC_NAMES)
    ]


def _save_cached(
    path: Path,
    scenario: ScenarioName,
    config: BenchConfig,
    runs: List[Dict[MetricName, float]],
) -> None:
    payload = {
        "scenario": scenario,
        "config": {
            "name": config.name,
            "N": config.N,
            "D": config.D,
            "M": config.M,
            "K": config.K,
            "n_list": config.n_list,
            "n_probe": config.n_probe,
        },
        "metrics": list(METRIC_NAMES),
        "runs": runs,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _compute_summary(
    runs: List[Dict[MetricName, float]],
) -> Dict[MetricName, Dict[str, float]]:
    if not runs:
        raise ValueError("No runs provided for summary")

    values = np.array(
        [[run[m] for m in METRIC_NAMES] for run in runs],
        dtype=float,
    )
    means = values.mean(axis=0)
    if len(runs) > 1:
        std = values.std(axis=0, ddof=1)
        ci95 = 1.96 * std / math.sqrt(len(runs))
    else:
        ci95 = np.zeros_like(means)

    summary: Dict[MetricName, Dict[str, float]] = {}
    for idx, metric in enumerate(METRIC_NAMES):
        summary[metric] = {
            "mean": float(means[idx]),
            "ci95": float(ci95[idx]),
        }
    return summary


def run_benchmarks(
    configs: List[BenchConfig],
    num_runs: int,
    force_recompute: bool = False,
) -> Dict[str, Dict[ScenarioName, Dict[MetricName, Dict[str, float]]]]:
    scenarios: Tuple[ScenarioName, ...] = ("fri_cluster", "fri_full", "merkle")

    summaries: Dict[str, Dict[ScenarioName, Dict[MetricName, Dict[str, float]]]] = {}

    for config in configs:
        config_summaries: Dict[ScenarioName, Dict[MetricName, Dict[str, float]]] = {}

        for scenario in scenarios:
            path = _result_file_name(scenario, config)

            runs: List[Dict[MetricName, float]] = []
            if path.exists() and not force_recompute:
                runs = _load_cached(path)

            existing_runs = len(runs)
            if existing_runs < num_runs:
                for _ in range(num_runs - existing_runs):
                    runs.append(_run_once(scenario, config))
                _save_cached(path, scenario, config, runs)

            config_summaries[scenario] = _compute_summary(runs)

        summaries[config.name] = config_summaries

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FRI vs Merkle commitment evaluation "
        "and save 95% CI summaries."
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of repetitions per configuration and scenario.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached results and recompute all benchmarks.",
    )

    args = parser.parse_args()

    summaries = run_benchmarks(
        configs=DEFAULT_CONFIGS,
        num_runs=args.num_runs,
        force_recompute=args.force_recompute,
    )

    for config_name, scenarios in summaries.items():
        print("=" * 80)
        print(f"Config: {config_name}")
        for scenario_name, metrics in scenarios.items():
            print(f"  Scenario: {scenario_name}")
            for metric, stats in metrics.items():
                mean = stats["mean"]
                ci95 = stats["ci95"]
                print(f"    {metric:12s}: {mean:.6f} ± {ci95:.6f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping plot generation.")
        return

    metric_labels: Dict[MetricName, str] = {
        "duration": "Duration (s)",
        "memory_peak": "Peak Memory (bytes)",
    }

    config_names = [cfg.name for cfg in DEFAULT_CONFIGS]
    x = np.arange(len(config_names))
    width = 0.25

    fig, axes = plt.subplots(1, len(METRIC_NAMES), figsize=(4 * len(METRIC_NAMES), 4))

    if len(METRIC_NAMES) == 1:
        axes = [axes]

    scenario_order: Tuple[ScenarioName, ...] = ("fri_cluster", "fri_full", "merkle")
    scenario_labels: Dict[ScenarioName, str] = {
        "fri_cluster": "FRI (cluster)",
        "fri_full": "FRI (full)",
        "merkle": "Merkle",
    }

    offsets = {
        "fri_cluster": -width,
        "fri_full": 0.0,
        "merkle": width,
    }

    for idx, metric in enumerate(METRIC_NAMES):
        ax = axes[idx]
        for scenario in scenario_order:
            means = [
                summaries[cfg.name][scenario][metric]["mean"] for cfg in DEFAULT_CONFIGS
            ]
            ci = [
                summaries[cfg.name][scenario][metric]["ci95"] for cfg in DEFAULT_CONFIGS
            ]
            ax.bar(
                x + offsets[scenario],
                means,
                width,
                yerr=ci,
                label=scenario_labels[scenario],
                capsize=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(config_names, rotation=45, ha="right")
        ax.set_title(metric_labels[metric])
        ax.set_yscale("log", base=10)

    axes[0].legend()
    fig.tight_layout()

    output_path = RESULT_DIR / "commit_eval_summary.pdf"
    fig.savefig(output_path, dpi=150)
    print(f"Saved commitment evaluation summary plot to {output_path}")


if __name__ == "__main__":
    main()
