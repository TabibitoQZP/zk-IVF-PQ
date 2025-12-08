"""
用于测试不同规模下circuit-only和set-based方案的性能差距
"""

import os

SINGLE_THREAD = True
if SINGLE_THREAD:
    os.environ["RAYON_NUM_THREADS"] = "1"

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
    "build_time",
    "prove_time",
    "verify_time",
    "proof_size",
    "memory_used",
    "num_gates",
]


RESULT_DIR = Path("data") / "bench_result"
if SINGLE_THREAD:
    RESULT_DIR = Path("data") / "single_bench_result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BASE_METRIC_NAMES: Tuple[MetricName, ...] = (
    "build_time",
    "prove_time",
    "verify_time",
    "proof_size",
    "memory_used",
)

EXTRA_METRIC_NAMES: Tuple[MetricName, ...] = ("num_gates",)

# 所有需要在 summary 中统计的指标
METRIC_NAMES: Tuple[MetricName, ...] = BASE_METRIC_NAMES + EXTRA_METRIC_NAMES

# 需要画图展示的指标（去掉 build_time, 新增 num_gates）
PLOT_METRICS: Tuple[MetricName, ...] = (
    "prove_time",
    "verify_time",
    "proof_size",
    "memory_used",
    "num_gates",
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
        merkled=False,
    ),
    BenchConfig(
        name="basic-merkle",  # 基础测试, 后面都是以这个为基础
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
        merkled=False,
    ),
    BenchConfig(
        name="low-acc-merkle",  # 超低精度测试, 主要是测试circuit是否有机会
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
    #     D=256,
    #     M=16,
    #     K=256,
    #     n_list=512,
    #     n_probe=64,
    #     top_k=128,
    #     merkled=False,
    # ),
    # BenchConfig(
    #     name="large-merkle",  # 大规模, 高精度测试
    #     N=65536,
    #     D=256,
    #     M=16,
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
        (
            build_time,
            prove_time,
            verify_time,
            proof_size,
            memory_used,
            num_gates,
        ) = set_bench(
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
        (
            build_time,
            prove_time,
            verify_time,
            proof_size,
            memory_used,
            num_gates,
        ) = circuit_bench(
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
        "num_gates": float(num_gates),
    }
    return result


def _load_cached(path: Path) -> List[Dict[MetricName, float]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    runs = payload.get("runs", [])
    parsed_runs: List[Dict[MetricName, float]] = []
    for run in runs:
        parsed: Dict[MetricName, float] = {}
        # 旧的 JSON 里可能没有 num_gates, 因此按需读取
        for metric in METRIC_NAMES:
            if metric in run:
                parsed[metric] = float(run[metric])
        # 至少需要基础五个指标都存在才认为这一条有效
        if all(m in parsed for m in BASE_METRIC_NAMES):
            parsed_runs.append(parsed)
    return parsed_runs


def _save_cached(
    path: Path,
    system: Literal["set_based", "circuit_based"],
    config: BenchConfig,
    runs: List[Dict[MetricName, float]],
) -> None:
    payload = {
        "system": system,
        "config": asdict(config),
        # metrics 字段主要用于标注基础指标；num_gates 直接存放在 runs 里
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

    summary: Dict[MetricName, Dict[str, float]] = {}
    for metric in METRIC_NAMES:
        values = [run[metric] for run in runs if metric in run]
        if not values:
            # 某个指标在当前 config / system 下完全不存在, 直接跳过
            continue
        arr = np.array(values, dtype=float)
        mean = float(arr.mean())
        if len(values) > 1:
            std = arr.std(ddof=1)
            ci95 = float(1.96 * std / math.sqrt(len(values)))
        else:
            ci95 = 0.0
        summary[metric] = {
            "mean": mean,
            "ci95": ci95,
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

                # 对存储空间相关的指标换算单位后打印
                if metric == "proof_size":
                    scale = 1.0 / 1024.0  # bytes -> kB
                    mean_disp = mean * scale
                    ci95_disp = ci95 * scale
                    unit = "kB"
                elif metric == "memory_used":
                    scale = 1.0 / (1024.0**3)  # bytes -> GiB
                    mean_disp = mean * scale
                    ci95_disp = ci95 * scale
                    unit = "GiB"
                else:
                    mean_disp = mean
                    ci95_disp = ci95
                    unit = ""

                if unit:
                    print(f"    {metric:12s}: {mean_disp:.4f} ± {ci95_disp:.4f} {unit}")
                else:
                    print(f"    {metric:12s}: {mean_disp:.4f} ± {ci95_disp:.4f}")

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
        "num_gates": "Number of Gates",
    }

    config_names = [cfg.name for cfg in DEFAULT_CONFIGS]
    x = np.arange(len(config_names))
    width = 0.35

    # 仅对可用的 PLOT_METRICS 画图；旧 JSON 中可能没有 num_gates
    available_metrics: List[MetricName] = []
    for metric in PLOT_METRICS:
        present = True
        for cfg in DEFAULT_CONFIGS:
            if (
                metric not in summaries[cfg.name]["set_based"]
                or metric not in summaries[cfg.name]["circuit_based"]
            ):
                present = False
                break
        if present:
            available_metrics.append(metric)

    if not available_metrics:
        print("No metrics available for plotting.")
        return

    fig, axes = plt.subplots(
        1, len(available_metrics), figsize=(4 * len(available_metrics), 4)
    )

    if len(available_metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(available_metrics):
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
            label="multi-set based",
            capsize=3,
        )
        ax.bar(
            x + width / 2,
            circuit_means,
            width,
            yerr=circuit_ci,
            label="circuit-only",
            capsize=3,
        )

        # 对 prove_time、memory_used 和 num_gates 使用 log10 轴
        if metric in ("prove_time", "memory_used", "num_gates"):
            ax.set_yscale("log")

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
