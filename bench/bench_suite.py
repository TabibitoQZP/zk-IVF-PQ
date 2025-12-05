"""
用于测试不同规模下circuit-only和set-based方案的性能差距
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np

from bench.circuit_based import bench as circuit_bench
from bench.set_based import bench as set_bench


MetricName = Literal[
    "build_time", "prove_time", "verify_time", "proof_size", "memory_used"
]


RESULT_DIR = Path("data") / "bench_result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

METRIC_NAMES: Tuple[MetricName, ...] = (
    "build_time",
    "prove_time",
    "verify_time",
    "proof_size",
    "memory_used",
)


@dataclass
class BenchConfig:
    name: str
    N: int
    D: int
    M: int
    K: int
    n_list: int
    n_probe: int
    top_k: int = 64
    merkled: bool = True

    @property
    def n(self) -> int:
        return self.N // self.n_list

    @property
    def d(self) -> int:
        return self.D // self.M


DEFAULT_CONFIGS: List[BenchConfig] = [
    BenchConfig(
        name="basic",  # 基础测试, 后面都是以这个为基础
        N=8192,
        D=128,
        M=8,
        K=16,
        n_list=256,
        n_probe=16,
        top_k=64,
        merkled=True,
    ),
    BenchConfig(
        name="low-acc",  # 超低精度测试, 主要是测试circuit是否有机会
        N=8192,
        D=128,
        M=8,
        K=1,
        n_list=16,
        n_probe=1,
        top_k=1,
        merkled=True,
    ),
    # BenchConfig(
    #     name="large",  # 大规模, 高精度测试
    #     N=65536,
    #     D=1024,
    #     M=32,
    #     K=256,
    #     n_list=512,
    #     n_probe=64,
    #     top_k=128,
    #     merkled=True,
    # ),
]


def _result_file_name(
    system: Literal["set_based", "circuit_based"], config: BenchConfig
) -> Path:
    name = (
        f"{system}"
        f"_N{config.N}_D{config.D}_M{config.M}_K{config.K}"
        f"_nlist{config.n_list}_nprobe{config.n_probe}"
        f"_topk{config.top_k}_merkled{int(config.merkled)}"
        f"_{config.name}.json"
    )
    return RESULT_DIR / name


def _run_once(
    system: Literal["set_based", "circuit_based"], config: BenchConfig
) -> Dict[MetricName, float]:
    if config.N % config.n_list != 0:
        raise ValueError(f"N must be divisible by n_list for config {config.name}")
    if config.D % config.M != 0:
        raise ValueError(f"D must be divisible by M for config {config.name}")

    if system == "set_based":
        build_time, prove_time, verify_time, proof_size, memory_used = set_bench(
            config.D,
            config.n_list,
            config.M,
            config.K,
            config.d,
            config.n_probe,
            config.n,
            config.top_k,
            config.merkled,
        )
    else:
        build_time, prove_time, verify_time, proof_size, memory_used = circuit_bench(
            config.D,
            config.n_list,
            config.M,
            config.K,
            config.d,
            config.n_probe,
            config.n,
            config.top_k,
            config.merkled,
        )

    result: Dict[MetricName, float] = {
        "build_time": float(build_time),
        "prove_time": float(prove_time),
        "verify_time": float(verify_time),
        "proof_size": float(proof_size),
        "memory_used": float(memory_used),
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
    system: Literal["set_based", "circuit_based"],
    config: BenchConfig,
    runs: List[Dict[MetricName, float]],
) -> None:
    payload = {
        "system": system,
        "config": asdict(config),
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
) -> Dict[str, Dict[str, Dict[MetricName, Dict[str, float]]]]:
    systems: Tuple[Literal["set_based", "circuit_based"], ...] = (
        "set_based",
        "circuit_based",
    )

    summaries: Dict[str, Dict[str, Dict[MetricName, Dict[str, float]]]] = {}

    for config in configs:
        config_summaries: Dict[str, Dict[MetricName, Dict[str, float]]] = {}

        for system in systems:
            path = _result_file_name(system, config)

            runs: List[Dict[MetricName, float]] = []
            if path.exists() and not force_recompute:
                runs = _load_cached(path)

            existing_runs = len(runs)
            if existing_runs < num_runs:
                for _ in range(num_runs - existing_runs):
                    runs.append(_run_once(system, config))
                _save_cached(path, system, config, runs)

            config_summaries[system] = _compute_summary(runs)

        summaries[config.name] = config_summaries

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark set-based vs circuit-based IVF-PQ proofs "
        "and save 95% CI summaries."
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of repetitions per configuration and system.",
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

    for config_name, systems in summaries.items():
        print("=" * 80)
        print(f"Config: {config_name}")
        for system_name, metrics in systems.items():
            print(f"  System: {system_name}")
            for metric, stats in metrics.items():
                mean = stats["mean"]
                ci95 = stats["ci95"]
                print(f"    {metric:12s}: {mean:.4f} ± {ci95:.4f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping plot generation.")
        return

    metric_labels: Dict[MetricName, str] = {
        "build_time": "Build Time (s)",
        "prove_time": "Prove Time (s)",
        "verify_time": "Verify Time (s)",
        "proof_size": "Proof Size (bytes)",
        "memory_used": "Peak Memory (bytes)",
    }

    config_names = [cfg.name for cfg in DEFAULT_CONFIGS]
    x = np.arange(len(config_names))
    width = 0.35

    fig, axes = plt.subplots(1, len(METRIC_NAMES), figsize=(4 * len(METRIC_NAMES), 4))

    if len(METRIC_NAMES) == 1:
        axes = [axes]

    for idx, metric in enumerate(METRIC_NAMES):
        ax = axes[idx]
        set_means = [
            summaries[cfg.name]["set_based"][metric]["mean"] for cfg in DEFAULT_CONFIGS
        ]
        set_ci = [
            summaries[cfg.name]["set_based"][metric]["ci95"] for cfg in DEFAULT_CONFIGS
        ]
        circuit_means = [
            summaries[cfg.name]["circuit_based"][metric]["mean"]
            for cfg in DEFAULT_CONFIGS
        ]
        circuit_ci = [
            summaries[cfg.name]["circuit_based"][metric]["ci95"]
            for cfg in DEFAULT_CONFIGS
        ]

        ax.bar(
            x - width / 2,
            set_means,
            width,
            yerr=set_ci,
            label="Set-based",
            capsize=3,
        )
        ax.bar(
            x + width / 2,
            circuit_means,
            width,
            yerr=circuit_ci,
            label="Circuit-based",
            capsize=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(config_names, rotation=45, ha="right")
        ax.set_title(metric_labels[metric])

    axes[0].legend()
    fig.tight_layout()

    output_path = RESULT_DIR / "bench_summary.pdf"
    fig.savefig(output_path, dpi=150)
    print(f"Saved benchmark summary plot to {output_path}")


if __name__ == "__main__":
    main()
