"""
Sweep IVF-PQ parameters to study how the set-based prove time (without Merkle
commitments) changes with the codebook size K, under a fixed
(N, D, n_list, n_probe, mem_bits) where

    mem_bits = M * log_2K

is kept constant, and we only sweep powers-of-two K.

We fix:
    - total number of vectors N,
    - dimension D,
    - number of IVF clusters n_list,
    - number of probed clusters n_probe,
    - mem_bits, which encodes the PQ code length per vector as mem_bits = M * log_2K.

We sweep:
    log_2K = 1, 2, 3, ..., c
    K      = 2**log_2K
and for each candidate we set:
    M      = mem_bits / log_2K
    n      = N // n_list
so that selected_count = n_probe * n remains fixed across the sweep.

For each configuration we run the set-based benchmark (Merkle-disabled)
multiple times, record the prove_time, compute the mean and 95% confidence
interval, and generate a line plot that visualizes the trend.
"""

from __future__ import annotations

import os

SINGLE_THREAD = False
if SINGLE_THREAD:
    os.environ["RAYON_NUM_THREADS"] = "1"


import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from bench.set_based import bench as set_bench


RESULT_DIR = Path("data") / "optimal_mem_config"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _result_file_path(
    N: int,
    D: int,
    n_list: int,
    n_probe: int,
    mem_bits: int,
    num_log_2K: int,
    num_runs: int,
) -> Path:
    return RESULT_DIR / (
        f"optimal_mem_N{N}_D{D}_nlist{n_list}_nprobe{n_probe}"
        f"_mem{mem_bits}_c{num_log_2K}_runs{num_runs}.json"
    )


@dataclass
class MemConfigResult:
    log_2K: int
    K: int
    M: int
    n_list: int
    n: int
    n_probe: int
    selected_count: int
    mem_bits: int
    prove_runs: List[float]
    num_gates_runs: List[float]

    @property
    def prove_mean(self) -> float:
        if not self.prove_runs:
            raise ValueError("No runs recorded for this configuration")
        arr = np.asarray(self.prove_runs, dtype=float)
        return float(arr.mean())

    @property
    def prove_ci95(self) -> float:
        if len(self.prove_runs) <= 1:
            return 0.0
        arr = np.asarray(self.prove_runs, dtype=float)
        std = float(arr.std(ddof=1))
        return float(1.96 * std / math.sqrt(len(self.prove_runs)))

    @property
    def num_gates_mean(self) -> float:
        if not self.num_gates_runs:
            raise ValueError("No num_gates recorded for this configuration")
        arr = np.asarray(self.num_gates_runs, dtype=float)
        return float(arr.mean())

    @property
    def num_gates_ci95(self) -> float:
        if len(self.num_gates_runs) <= 1:
            return 0.0
        arr = np.asarray(self.num_gates_runs, dtype=float)
        std = float(arr.std(ddof=1))
        return float(1.96 * std / math.sqrt(len(self.num_gates_runs)))


def _generate_log_2K_candidates(num_log_2K: int) -> List[int]:
    if num_log_2K <= 0:
        raise ValueError("num_log_2K (c) must be positive")
    return list(range(1, num_log_2K + 1))


def _compute_n_and_selected_count(
    N: int,
    n_list: int,
    n_probe: int,
) -> Tuple[int, int]:
    if N <= 0:
        raise ValueError("N must be positive")
    if n_list <= 0:
        raise ValueError("n_list must be positive")
    if N % n_list != 0:
        raise ValueError(f"N={N} must be divisible by n_list={n_list}")
    if n_probe <= 0:
        raise ValueError("n_probe must be positive")
    if n_probe > n_list:
        raise ValueError(f"n_probe={n_probe} must not exceed n_list={n_list}")

    n = N // n_list
    selected_count = n_probe * n
    return n, selected_count


def _compute_M_for_constant_mem(mem_bits: int, log_2K: int, D: int) -> int:
    if mem_bits <= 0:
        raise ValueError("mem_bits must be positive")
    if log_2K <= 0:
        raise ValueError("log_2K must be positive")

    if mem_bits % log_2K != 0:
        raise ValueError(
            f"mem_bits={mem_bits} must be divisible by log_2K={log_2K} "
            "so that M is an integer."
        )

    M = mem_bits // log_2K
    if M <= 0:
        raise ValueError(
            f"Derived M={M} must be positive for mem_bits={mem_bits}, log_2K={log_2K}."
        )
    if D % M != 0:
        raise ValueError(
            f"D={D} must be divisible by M={M} (mem_bits={mem_bits}, log_2K={log_2K})."
        )
    return M


def _run_once(
    N: int,
    D: int,
    n_list: int,
    n_probe: int,
    mem_bits: int,
    log_2K: int,
) -> Tuple[float, float, int, int]:
    K = 1 << log_2K
    M = _compute_M_for_constant_mem(mem_bits=mem_bits, log_2K=log_2K, D=D)
    d = D // M
    n, selected_count = _compute_n_and_selected_count(N=N, n_list=n_list, n_probe=n_probe)

    (
        build_time,
        prove_time,
        verify_time,
        proof_size,
        memory_used,
        num_gates,
    ) = set_bench(D, n_list, M, K, d, n_probe, n, merkled=False)

    return float(prove_time), float(num_gates), M, selected_count


def sweep_configs(
    N: int,
    D: int,
    n_list: int,
    n_probe: int,
    mem_bits: int,
    num_log_2K: int,
    num_runs: int,
) -> Dict[int, MemConfigResult]:
    if num_runs <= 0:
        raise ValueError("num_runs must be positive")

    n, selected_count = _compute_n_and_selected_count(N=N, n_list=n_list, n_probe=n_probe)
    log_2K_candidates = _generate_log_2K_candidates(num_log_2K=num_log_2K)

    results: Dict[int, MemConfigResult] = {}
    for log_2K in log_2K_candidates:
        K = 1 << log_2K

        try:
            M = _compute_M_for_constant_mem(mem_bits=mem_bits, log_2K=log_2K, D=D)
        except ValueError as e:
            print(f"Skipping (log_2K={log_2K}, K={K}): {e}")
            continue

        prove_runs: List[float] = []
        num_gates_runs: List[float] = []

        print(
            f"Running config: log_2K={log_2K}, K={K}, M={M}, mem_bits={mem_bits}, "
            f"N={N}, D={D}, n_list={n_list}, n={n}, n_probe={n_probe}, "
            f"selected_count={selected_count}"
        )
        for run_idx in range(num_runs):
            print(f"  Run {run_idx + 1}/{num_runs} ...")
            prove_time, num_gates, _M, _sel = _run_once(
                N=N,
                D=D,
                n_list=n_list,
                n_probe=n_probe,
                mem_bits=mem_bits,
                log_2K=log_2K,
            )
            prove_runs.append(prove_time)
            num_gates_runs.append(num_gates)

        results[K] = MemConfigResult(
            log_2K=log_2K,
            K=K,
            M=M,
            n_list=n_list,
            n=n,
            n_probe=n_probe,
            selected_count=selected_count,
            mem_bits=mem_bits,
            prove_runs=prove_runs,
            num_gates_runs=num_gates_runs,
        )

    if not results:
        raise ValueError(
            "No valid (log_2K, K, M) configurations found for the given mem_bits and D."
        )
    return results


def _select_optimal_config(results: Dict[int, MemConfigResult]) -> MemConfigResult:
    if not results:
        raise ValueError("No results to select from")
    return min(results.values(), key=lambda r: r.prove_mean)


def _load_cached_results(path: Path) -> Dict[int, MemConfigResult]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_results = payload.get("results", {})
    results: Dict[int, MemConfigResult] = {}
    for key, cfg in raw_results.items():
        K = int(cfg.get("K", key))
        results[K] = MemConfigResult(
            log_2K=int(cfg["log_2K"]),
            K=K,
            M=int(cfg["M"]),
            n_list=int(cfg["n_list"]),
            n=int(cfg["n"]),
            n_probe=int(cfg["n_probe"]),
            selected_count=int(cfg["selected_count"]),
            mem_bits=int(cfg["mem_bits"]),
            prove_runs=[float(v) for v in cfg.get("prove_runs", [])],
            num_gates_runs=[float(v) for v in cfg.get("num_gates_runs", [])],
        )
    return results


def _save_results(
    N: int,
    D: int,
    n_list: int,
    n_probe: int,
    mem_bits: int,
    num_log_2K: int,
    num_runs: int,
    results: Dict[int, MemConfigResult],
) -> Path:
    n, selected_count = _compute_n_and_selected_count(N=N, n_list=n_list, n_probe=n_probe)
    payload = {
        "N": N,
        "D": D,
        "n_list": n_list,
        "n_probe": n_probe,
        "n": n,
        "selected_count": selected_count,
        "mem_bits": mem_bits,
        "num_log_2K": num_log_2K,
        "num_runs": num_runs,
        "results": {
            K: {
                "log_2K": res.log_2K,
                "K": res.K,
                "M": res.M,
                "n_list": res.n_list,
                "n": res.n,
                "n_probe": res.n_probe,
                "selected_count": res.selected_count,
                "mem_bits": res.mem_bits,
                "prove_runs": res.prove_runs,
                "prove_mean": res.prove_mean,
                "prove_ci95": res.prove_ci95,
                "num_gates_runs": res.num_gates_runs,
                "num_gates_mean": res.num_gates_mean,
                "num_gates_ci95": res.num_gates_ci95,
            }
            for K, res in results.items()
        },
    }

    output_path = _result_file_path(
        N=N,
        D=D,
        n_list=n_list,
        n_probe=n_probe,
        mem_bits=mem_bits,
        num_log_2K=num_log_2K,
        num_runs=num_runs,
    )
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def _plot_results(
    N: int,
    D: int,
    n_list: int,
    n_probe: int,
    selected_count: int,
    mem_bits: int,
    num_log_2K: int,
    results: Dict[int, MemConfigResult],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping plot generation.")
        return

    Ks = sorted(results.keys())
    prove_means = [results[K].prove_mean for K in Ks]
    prove_ci95 = [results[K].prove_ci95 for K in Ks]
    num_gates_means = [results[K].num_gates_mean for K in Ks]
    num_gates_ci95 = [results[K].num_gates_ci95 for K in Ks]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.errorbar(
        Ks,
        prove_means,
        yerr=prove_ci95,
        fmt="-o",
        capsize=3,
        label="Prove time (mean ± 95% CI)",
        color="C0",
    )

    ax2 = ax1.twinx()
    ax2.errorbar(
        Ks,
        num_gates_means,
        yerr=num_gates_ci95,
        fmt="-s",
        capsize=3,
        label="Num gates (mean ± 95% CI)",
        color="C1",
    )

    ax1.set_xlabel("K (codebook size per sub-vector)")
    ax1.set_ylabel("Prove time (seconds)", color="C0")
    ax2.set_ylabel("Number of gates", color="C1")
    ax1.set_title(
        "Set-based IVF-PQ (no Merkle) prove time & num_gates vs K\n"
        f"N={N}, D={D}, n_list={n_list}, n_probe={n_probe}, "
        f"selected_count={selected_count}, mem_bits={mem_bits}, c={num_log_2K}"
    )
    ax1.set_xscale("log", base=2)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()

    output_path = RESULT_DIR / (
        f"optimal_mem_N{N}_D{D}_nlist{n_list}_nprobe{n_probe}"
        f"_mem{mem_bits}_c{num_log_2K}.pdf"
    )
    fig.savefig(output_path, dpi=150)
    print(f"Saved optimal-mem-config plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep K for the set-based IVF-PQ scheme without Merkle commitments, "
            "sweeping powers-of-two K where log_2K starts from 1 and increments by 1, "
            "and adjusting M so that mem_bits = M * log_2K remains constant."
        )
    )
    parser.add_argument("--N", type=int, default=2**16, help="Total number of vectors.")
    parser.add_argument("--D", type=int, default=128, help="Vector dimension.")
    parser.add_argument(
        "--n-list",
        type=int,
        default=256,
        help="Number of IVF clusters (kept fixed across K sweep).",
    )
    parser.add_argument(
        "--n-probe",
        type=int,
        default=16,
        help="Number of probed IVF clusters per query (kept fixed across K sweep).",
    )
    parser.add_argument(
        "--mem-bits",
        type=int,
        default=32,
        help="PQ code length in bits: mem_bits = M * log_2K (kept fixed across the sweep).",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=10,
        help="Number of K values to test (log_2K = 1..c, K = 2**log_2K).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of repetitions per configuration.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached results and recompute all configurations.",
    )

    args = parser.parse_args()

    N = args.N
    D = args.D
    n_list = args.n_list
    n_probe = args.n_probe
    mem_bits = args.mem_bits
    num_log_2K = args.c
    num_runs = args.num_runs

    n, selected_count = _compute_n_and_selected_count(N=N, n_list=n_list, n_probe=n_probe)

    json_path = _result_file_path(
        N=N,
        D=D,
        n_list=n_list,
        n_probe=n_probe,
        mem_bits=mem_bits,
        num_log_2K=num_log_2K,
        num_runs=num_runs,
    )

    if json_path.exists() and not args.force_recompute:
        print(f"Loading cached results from {json_path}")
        results = _load_cached_results(json_path)
    else:
        results = sweep_configs(
            N=N,
            D=D,
            n_list=n_list,
            n_probe=n_probe,
            mem_bits=mem_bits,
            num_log_2K=num_log_2K,
            num_runs=num_runs,
        )
        json_path = _save_results(
            N=N,
            D=D,
            n_list=n_list,
            n_probe=n_probe,
            mem_bits=mem_bits,
            num_log_2K=num_log_2K,
            num_runs=num_runs,
            results=results,
        )
        print(f"Saved raw results to {json_path}")

    optimal = _select_optimal_config(results)

    print("\nSummary of configurations (prove_time / num_gates mean ± 95% CI):")
    for K in sorted(results.keys()):
        res = results[K]
        print(
            f"  log_2K={res.log_2K:2d}, K={res.K:6d}, M={res.M:6d} -> "
            f"prove_time={res.prove_mean:.4f} ± {res.prove_ci95:.4f} s, "
            f"num_gates={res.num_gates_mean:.2f} ± {res.num_gates_ci95:.2f}"
        )

    print("\nFixed query-time candidate count:")
    print(f"  n={n}, n_probe={n_probe} => selected_count={selected_count}")

    print("\nEmpirically optimal configuration (by mean prove_time):")
    print(
        f"  log_2K={optimal.log_2K}, K={optimal.K}, M={optimal.M}, "
        f"mean prove_time={optimal.prove_mean:.4f} s ± {optimal.prove_ci95:.4f} s, "
        f"num_gates={optimal.num_gates_mean:.2f} ± {optimal.num_gates_ci95:.2f}"
    )

    _plot_results(
        N=N,
        D=D,
        n_list=n_list,
        n_probe=n_probe,
        selected_count=selected_count,
        mem_bits=mem_bits,
        num_log_2K=num_log_2K,
        results=results,
    )


if __name__ == "__main__":
    main()

