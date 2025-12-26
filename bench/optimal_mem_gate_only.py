"""
Gate-only sweep over (B, K) for IVF-PQ set-based scheme.

This script mirrors the parameterization in `bench_free_bench/set_based_gate.py`,
but:
  - takes (D, n_list, n_probe, n) from CLI,
  - uses fixed, hard-coded search grids for B and K,
  - caches results to avoid re-running identical configurations,
  - plots a line chart: x-axis K, y-axis num_gates, one line per B.

We use the Rust-exposed function:
    py_set_based_gate(M, K, d, n_list, n_probe, n, top_k, merkled) -> int
where:
    B        = M * log2(K)
    log2(K)  = log_2K
    M        = B / log_2K (must be integer)
    d        = D / M      (must be integer)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


RESULT_DIR = Path("data") / "optimal_mem_gate_only"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Default search grids (CLI can override).
DEFAULT_B_VALUES: List[int] = [8, 16, 32, 64, 128]
DEFAULT_LOG2K_VALUES: List[int] = [1, 2, 4, 8]


@dataclass(frozen=True)
class GateConfig:
    D: int
    n_list: int
    n_probe: int
    n: int
    top_k: int
    merkled: bool
    B: int
    log_2K: int

    @property
    def K(self) -> int:
        return 1 << self.log_2K

    @property
    def M(self) -> int:
        return self.B // self.log_2K

    @property
    def d(self) -> int:
        # If D is not divisible by M, the last sub-vector is padded, so we use
        # ceil(D / M) instead of floor(D / M).
        return (self.D + self.M - 1) // self.M


def _cache_path(D: int, n_list: int, n_probe: int, n: int, top_k: int, merkled: bool) -> Path:
    return RESULT_DIR / (
        f"gate_only_D{D}_nlist{n_list}_nprobe{n_probe}_n{n}_topk{top_k}"
        f"_merkled{int(merkled)}.json"
    )


def _load_cache(path: Path) -> Tuple[Dict[Tuple[int, int], int], dict]:
    if not path.exists():
        return {}, {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    results: Dict[Tuple[int, int], int] = {}
    raw = payload.get("results", {})
    for b_str, ks in raw.items():
        B = int(b_str)
        if not isinstance(ks, dict):
            continue
        for k_str, record in ks.items():
            K = int(k_str)
            if isinstance(record, dict) and "num_gates" in record:
                results[(B, K)] = int(record["num_gates"])
            elif isinstance(record, (int, float)):
                results[(B, K)] = int(record)
    return results, payload


def _save_cache(
    path: Path,
    *,
    D: int,
    n_list: int,
    n_probe: int,
    n: int,
    top_k: int,
    merkled: bool,
    B_values: List[int],
    log2k_values: List[int],
    results: Dict[Tuple[int, int], int],
    skipped: List[dict],
) -> None:
    nested: Dict[str, Dict[str, dict]] = {}
    for (B, K), gates in sorted(results.items()):
        nested.setdefault(str(B), {})[str(K)] = {"num_gates": int(gates)}

    payload = {
        "D": D,
        "n_list": n_list,
        "n_probe": n_probe,
        "n": n,
        "top_k": top_k,
        "merkled": bool(merkled),
        "grid": {"B_values": B_values, "log_2K_values": log2k_values},
        "results": nested,
        "skipped": skipped,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _validate_config(cfg: GateConfig) -> Optional[str]:
    if cfg.D <= 0:
        return "D must be positive"
    if cfg.n_list <= 0:
        return "n_list must be positive"
    if cfg.n_probe <= 0:
        return "n_probe must be positive"
    if cfg.n_probe > cfg.n_list:
        return "n_probe must not exceed n_list"
    if cfg.n <= 0:
        return "n must be positive"
    if cfg.top_k <= 0:
        return "top_k must be positive"
    if cfg.B <= 0:
        return "B must be positive"
    if cfg.log_2K <= 0:
        return "log_2K must be positive"
    if cfg.B % cfg.log_2K != 0:
        return f"B={cfg.B} must be divisible by log_2K={cfg.log_2K}"
    if cfg.M <= 0:
        return f"Derived M={cfg.M} must be positive"
    if cfg.M > cfg.D:
        return f"Derived M={cfg.M} must not exceed D={cfg.D}"
    return None


def _compute_num_gates(cfg: GateConfig) -> int:
    try:
        from zk_IVF_PQ.zk_IVF_PQ import py_set_based_gate
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import `py_set_based_gate` from `zk_IVF_PQ.zk_IVF_PQ`. "
            "Build/install the Python extension first (e.g. `maturin develop`)."
        ) from e

    err = _validate_config(cfg)
    if err is not None:
        raise ValueError(err)

    return int(
        py_set_based_gate(
            cfg.M,
            cfg.K,
            cfg.d,
            cfg.n_list,
            cfg.n_probe,
            cfg.n,
            cfg.top_k,
            cfg.merkled,
        )
    )


def _plot_results(
    *,
    D: int,
    n_list: int,
    n_probe: int,
    n: int,
    top_k: int,
    merkled: bool,
    results: Dict[Tuple[int, int], int],
    B_values: List[int],
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("matplotlib is not available; skipping plot generation.")
        return None

    fig, ax = plt.subplots(figsize=(7, 4))

    for idx, B in enumerate(B_values):
        points = sorted((K, gates) for ((b, K), gates) in results.items() if b == B)
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"B={B}", color=f"C{idx}")

    ax.set_xlabel("K (codebook size per sub-vector)")
    ax.set_ylabel("Number of gates")
    ax.set_title(
        "Set-based IVF-PQ num_gates vs K (gate-only)\n"
        f"D={D}, n_list={n_list}, n_probe={n_probe}, n={n}, top_k={top_k}, merkled={int(merkled)}"
    )
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(loc="best")
    fig.tight_layout()

    out = RESULT_DIR / (
        f"gate_only_D{D}_nlist{n_list}_nprobe{n_probe}_n{n}_topk{top_k}"
        f"_merkled{int(merkled)}.pdf"
    )
    fig.savefig(out, dpi=150)
    print(f"Saved plot to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gate-only sweep for set-based IVF-PQ: compute num_gates over fixed (B, K) grids, "
            "cache per configuration, and plot num_gates vs K (one line per B)."
        )
    )
    parser.add_argument("--D", type=int, default=960, help="Vector dimension.")
    parser.add_argument("--n-list", type=int, default=8192, help="Number of IVF clusters.")
    parser.add_argument("--n-probe", type=int, default=64, help="Number of probed clusters.")
    parser.add_argument("--n", type=int, default=256, help="Number of points per cluster.")
    parser.add_argument("--top-k", type=int, default=64, help="Top-k used in the gadget.")
    parser.add_argument(
        "--B-values",
        type=str,
        default=",".join(str(v) for v in DEFAULT_B_VALUES),
        help="Comma-separated grid for B (default: 8,16,32,64).",
    )
    parser.add_argument(
        "--log2k-values",
        type=str,
        default=",".join(str(v) for v in DEFAULT_LOG2K_VALUES),
        help="Comma-separated grid for log2(K) (default: 1,2,4,8).",
    )
    parser.add_argument(
        "--merkled",
        action="store_true",
        help="Enable Merkle commitments in the gate count.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached results and recompute all configurations.",
    )
    args = parser.parse_args()

    D = args.D
    n_list = args.n_list
    n_probe = args.n_probe
    n = args.n
    top_k = args.top_k
    merkled = bool(args.merkled)

    def _parse_int_list(s: str, *, name: str) -> List[int]:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if not parts:
            raise ValueError(f"{name} must be a non-empty comma-separated list")
        values: List[int] = []
        for p in parts:
            try:
                v = int(p)
            except ValueError as e:
                raise ValueError(f"Invalid integer in {name}: {p!r}") from e
            if v <= 0:
                raise ValueError(f"All values in {name} must be positive; got {v}")
            values.append(v)
        # Keep stable order but drop duplicates.
        deduped = list(dict.fromkeys(values).keys())
        return deduped

    B_values = _parse_int_list(args.B_values, name="--B-values")
    log2k_values = _parse_int_list(args.log2k_values, name="--log2k-values")

    cache_path = _cache_path(D=D, n_list=n_list, n_probe=n_probe, n=n, top_k=top_k, merkled=merkled)
    cached, payload = _load_cache(cache_path)
    skipped: List[dict] = list(payload.get("skipped", [])) if isinstance(payload, dict) else []

    if cache_path.exists() and not args.force_recompute:
        print(f"Loading cache from {cache_path}")
    else:
        cached = {}
        skipped = []

    results: Dict[Tuple[int, int], int] = dict(cached)

    for B in B_values:
        for log_2K in log2k_values:
            K = 1 << log_2K
            key = (B, K)
            if key in results and not args.force_recompute:
                continue

            cfg = GateConfig(
                D=D,
                n_list=n_list,
                n_probe=n_probe,
                n=n,
                top_k=top_k,
                merkled=merkled,
                B=B,
                log_2K=log_2K,
            )

            err = _validate_config(cfg)
            if err is not None:
                skipped.append({"B": B, "log_2K": log_2K, "K": K, "reason": err})
                continue

            gates = _compute_num_gates(cfg)
            results[key] = gates
            print(f"B={B:3d}, log2K={log_2K:2d}, K={K:6d}, M={cfg.M:4d}, d={cfg.d:4d} -> gates={gates}")

            _save_cache(
                cache_path,
                D=D,
                n_list=n_list,
                n_probe=n_probe,
                n=n,
                top_k=top_k,
                merkled=merkled,
                B_values=B_values,
                log2k_values=log2k_values,
                results=results,
                skipped=skipped,
            )

    _save_cache(
        cache_path,
        D=D,
        n_list=n_list,
        n_probe=n_probe,
        n=n,
        top_k=top_k,
        merkled=merkled,
        B_values=B_values,
        log2k_values=log2k_values,
        results=results,
        skipped=skipped,
    )
    print(f"Saved cache to {cache_path}")

    print("\nSummary (num_gates):")
    for B in B_values:
        points = sorted(
            (K, results[(B, K)])
            for log_2K in log2k_values
            for K in [1 << log_2K]
            if (B, K) in results
        )
        if not points:
            continue
        pretty = ", ".join(f"K={K}:{gates}" for K, gates in points)
        print(f"  B={B}: {pretty}")

    _plot_results(
        D=D,
        n_list=n_list,
        n_probe=n_probe,
        n=n,
        top_k=top_k,
        merkled=merkled,
        results=results,
        B_values=B_values,
    )


if __name__ == "__main__":
    main()
