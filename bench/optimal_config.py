"""
Sweep IVF-PQ parameters to study how the set-based prove time (without Merkle
commitments) changes with the number of IVF clusters n_list, under a fixed
(N, D, M, selected_count).

We fix:
    - total number of vectors N,
    - dimension D,
    - number of sub-vectors M,
    - selected_count = n_probe * n (total number of candidate points).

For a given N and selected_count, we consider
    n_list in {
        N // selected_count,
        (N // selected_count) * 2,
        (N // selected_count) * 4,
        ...,
        (N // selected_count) * 2**(c - 1),
    }
and for each candidate set
    n = N // n_list
    K = N // n_list
    n_probe = selected_count // n
so that selected_count = n_probe * n holds for every configuration.

For each configuration we run the set-based benchmark (Merkle-disabled)
multiple times,
record the prove_time, compute the mean and 95% confidence interval,
and generate a line plot that visualizes the trend of the generation time.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from bench.set_based import bench as set_bench


RESULT_DIR = Path("data") / "optimal_config"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ConfigResult:
    n_list: int
    n: int
    K: int
    n_probe: int
    runs: List[float]

    @property
    def mean(self) -> float:
        if not self.runs:
            raise ValueError("No runs recorded for this configuration")
        arr = np.asarray(self.runs, dtype=float)
        return float(arr.mean())

    @property
    def ci95(self) -> float:
        if len(self.runs) <= 1:
            return 0.0
        arr = np.asarray(self.runs, dtype=float)
        std = float(arr.std(ddof=1))
        return float(1.96 * std / math.sqrt(len(self.runs)))


def _generate_n_list_candidates(
    N: int,
    selected_count: int,
    num_n_list: int,
) -> List[int]:
    if N <= 0:
        raise ValueError("N must be positive")
    if selected_count <= 0:
        raise ValueError("selected_count must be positive")
    if selected_count > N:
        raise ValueError("selected_count must not exceed N")
    if num_n_list <= 0:
        raise ValueError("num_n_list (c) must be positive")

    if N % selected_count != 0:
        raise ValueError(
            f"selected_count={selected_count} must divide N={N} so that "
            "N // selected_count is an integer base n_list."
        )

    candidates: List[int] = []
    base_n_list = N // selected_count
    curr = base_n_list

    for _ in range(num_n_list):
        if curr > N:
            raise ValueError(
                f"Generated n_list={curr} exceeds N={N}; "
                "decrease c or increase selected_count."
            )
        if N % curr != 0:
            raise ValueError(f"N={N} must be divisible by n_list={curr}")
        # Ensure that selected_count can be written as n_probe * n
        n = N // curr
        if selected_count % n != 0:
            raise ValueError(
                f"For n_list={curr}, n={n} but selected_count={selected_count} "
                "is not divisible by n; cannot enforce selected_count = n_probe * n "
                "with integer n_probe."
            )
        candidates.append(curr)
        curr *= 2
    return candidates


def _compute_n_and_n_probe(
    N: int,
    selected_count: int,
    n_list: int,
) -> Tuple[int, int]:
    if N % n_list != 0:
        raise ValueError(f"N={N} must be divisible by n_list={n_list}")

    n = N // n_list
    if selected_count % n != 0:
        raise ValueError(
            f"selected_count={selected_count} must be divisible by n={n} "
            f"for n_list={n_list} so that selected_count = n_probe * n."
        )

    n_probe = selected_count // n
    if n_probe <= 0:
        raise ValueError(
            f"Derived n_probe={n_probe} must be positive "
            f"for n_list={n_list}, n={n}, selected_count={selected_count}."
        )
    if n_probe > n_list:
        raise ValueError(
            f"Derived n_probe={n_probe} must not exceed n_list={n_list}; "
            f"(selected_count={selected_count}, n={n}, N={N})."
        )
    return n, n_probe


def _run_once(
    N: int,
    D: int,
    M: int,
    selected_count: int,
    n_list: int,
) -> float:
    if D % M != 0:
        raise ValueError(f"D={D} must be divisible by M={M}")

    d = D // M
    n, n_probe = _compute_n_and_n_probe(N, selected_count, n_list)
    K = N // n_list

    (
        build_time,
        prove_time,
        verify_time,
        proof_size,
        memory_used,
    ) = set_bench(D, n_list, M, K, d, n_probe, n, merkled=False)

    # We treat prove_time as the "generation time" of the proof.
    _ = build_time, verify_time, proof_size, memory_used
    return float(prove_time)


def sweep_configs(
    N: int,
    D: int,
    M: int,
    selected_count: int,
    num_n_list: int,
    num_runs: int,
) -> Dict[int, ConfigResult]:
    if num_runs <= 0:
        raise ValueError("num_runs must be positive")

    n_list_candidates = _generate_n_list_candidates(
        N=N,
        selected_count=selected_count,
        num_n_list=num_n_list,
    )
    results: Dict[int, ConfigResult] = {}

    for n_list in n_list_candidates:
        n, n_probe = _compute_n_and_n_probe(N, selected_count, n_list)
        K = N // n_list
        runs: List[float] = []

        print(
            f"Running config: n_list={n_list}, n={n}, K={K}, "
            f"N={N}, D={D}, M={M}, n_probe={n_probe}, "
            f"selected_count={selected_count}"
        )
        for run_idx in range(num_runs):
            print(f"  Run {run_idx + 1}/{num_runs} ...")
            prove_time = _run_once(N, D, M, selected_count, n_list)
            runs.append(prove_time)

        results[n_list] = ConfigResult(
            n_list=n_list,
            n=n,
            K=K,
            n_probe=n_probe,
            runs=runs,
        )

    return results


def _select_optimal_config(results: Dict[int, ConfigResult]) -> ConfigResult:
    if not results:
        raise ValueError("No results to select from")
    # Choose the configuration with the minimal mean prove_time.
    return min(results.values(), key=lambda r: r.mean)


def _save_results(
    N: int,
    D: int,
    M: int,
    selected_count: int,
    num_n_list: int,
    num_runs: int,
    results: Dict[int, ConfigResult],
) -> Path:
    payload = {
        "N": N,
        "D": D,
        "M": M,
        "selected_count": selected_count,
        "num_n_list": num_n_list,
        "num_runs": num_runs,
        "results": {
            n_list: {
                "n_list": res.n_list,
                "n": res.n,
                "K": res.K,
                "n_probe": res.n_probe,
                "runs": res.runs,
                "mean": res.mean,
                "ci95": res.ci95,
            }
            for n_list, res in results.items()
        },
    }

    output_path = RESULT_DIR / (
        f"optimal_N{N}_D{D}_M{M}_sel{selected_count}_c{num_n_list}_runs{num_runs}.json"
    )
    with output_path.open("w", encoding="utf-8") as f:
        import json

        json.dump(payload, f, indent=2)

    return output_path


def _plot_results(
    N: int,
    D: int,
    M: int,
    selected_count: int,
    num_n_list: int,
    results: Dict[int, ConfigResult],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping plot generation.")
        return

    n_lists = sorted(results.keys())
    means = [results[n_list].mean for n_list in n_lists]
    ci95 = [results[n_list].ci95 for n_list in n_lists]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        n_lists,
        means,
        yerr=ci95,
        fmt="-o",
        capsize=3,
        label="Prove time (mean ± 95% CI)",
    )

    ax.set_xlabel("n_list (number of IVF clusters)")
    ax.set_ylabel("Prove time (seconds)")
    ax.set_title(
        f"Set-based IVF-PQ (no Merkle) prove time vs n_list\n"
        f"N={N}, D={D}, M={M}, selected_count={selected_count}, c={num_n_list}"
    )
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()
    fig.tight_layout()

    output_path = RESULT_DIR / (
        f"optimal_N{N}_D{D}_M{M}_sel{selected_count}_c{num_n_list}.pdf"
    )
    fig.savefig(output_path, dpi=150)
    print(f"Saved optimal-config plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep n_list (and corresponding n, K = N // n_list) for the "
            "set-based IVF-PQ scheme without Merkle commitments, run multiple "
            "times, compute mean prove time and 95% CI, and select the "
            "empirically optimal config."
        )
    )
    parser.add_argument("--N", type=int, default=1024, help="Total number of vectors.")
    parser.add_argument("--D", type=int, default=128, help="Vector dimension.")
    parser.add_argument(
        "--M",
        type=int,
        default=8,
        help="Number of sub-vectors (product quantization parameter).",
    )
    parser.add_argument(
        "--selected_count",
        type=int,
        default=512,
        help=(
            "Total number of candidate points selected per query, "
            "i.e., selected_count = n_probe * n."
        ),
    )
    parser.add_argument(
        "--c",
        type=int,
        default=5,
        help=(
            "Number of different n_list values to test "
            "(powers of two starting from N // selected_count)."
        ),
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of repetitions per configuration.",
    )

    args = parser.parse_args()

    N = args.N
    D = args.D
    M = args.M
    selected_count = args.selected_count
    num_n_list = args.c
    num_runs = args.num_runs

    results = sweep_configs(
        N=N,
        D=D,
        M=M,
        selected_count=selected_count,
        num_n_list=num_n_list,
        num_runs=num_runs,
    )
    optimal = _select_optimal_config(results)

    print("\nSummary of configurations (prove_time mean ± 95% CI):")
    for n_list in sorted(results.keys()):
        res = results[n_list]
        print(
            f"  n_list={res.n_list:6d}, n={res.n:6d}, K={res.K:6d}, "
            f"n_probe={res.n_probe:6d} -> "
            f"{res.mean:.4f} ± {res.ci95:.4f} s"
        )

    print("\nEmpirically optimal configuration (by mean prove_time):")
    print(
        f"  n_list={optimal.n_list}, n={optimal.n}, K={optimal.K}, "
        f"n_probe={optimal.n_probe}, "
        f"mean prove_time={optimal.mean:.4f} s ± {optimal.ci95:.4f} s"
    )

    json_path = _save_results(
        N=N,
        D=D,
        M=M,
        selected_count=selected_count,
        num_n_list=num_n_list,
        num_runs=num_runs,
        results=results,
    )
    print(f"Saved raw results to {json_path}")

    _plot_results(
        N=N,
        D=D,
        M=M,
        selected_count=selected_count,
        num_n_list=num_n_list,
        results=results,
    )


if __name__ == "__main__":
    main()
