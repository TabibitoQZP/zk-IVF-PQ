"""
Compute the set-based IVF-PQ gate count for a single configuration.

This is the non-sweeping counterpart of `bench.optimal_mem_gate_only`.
It accepts either:
  1. `B` + `log2K`, or
  2. `M` + `K`
and computes the corresponding `num_gates` once.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


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
    M: int
    K: int
    d: int


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _build_config(args: argparse.Namespace) -> GateConfig:
    for name in ("D", "n_list", "n_probe", "n", "top_k"):
        _validate_positive(name, getattr(args, name))

    if args.n_probe > args.n_list:
        raise ValueError(
            f"n_probe={args.n_probe} must not exceed n_list={args.n_list}"
        )

    has_b_form = args.B is not None or args.log2_k is not None
    has_m_form = args.M is not None or args.K is not None

    if has_b_form and has_m_form:
        raise ValueError("Specify either (B, log2K) or (M, K), not both")
    if not has_b_form and not has_m_form:
        raise ValueError("You must specify either (B, log2K) or (M, K)")

    if has_b_form:
        if args.B is None or args.log2_k is None:
            raise ValueError("Both --B and --log2K are required together")
        _validate_positive("B", args.B)
        _validate_positive("log2K", args.log2_k)
        if args.B % args.log2_k != 0:
            raise ValueError(
                f"B={args.B} must be divisible by log2K={args.log2_k}"
            )

        B = args.B
        log_2K = args.log2_k
        M = B // log_2K
        K = 1 << log_2K
    else:
        if args.M is None or args.K is None:
            raise ValueError("Both --M and --K are required together")
        _validate_positive("M", args.M)
        _validate_positive("K", args.K)
        if not _is_power_of_two(args.K):
            raise ValueError(f"K={args.K} must be a power of two")

        M = args.M
        K = args.K
        log_2K = args.K.bit_length() - 1
        B = M * log_2K

    if M > args.D:
        raise ValueError(f"Derived M={M} must not exceed D={args.D}")

    d = _ceil_div(args.D, M)

    return GateConfig(
        D=args.D,
        n_list=args.n_list,
        n_probe=args.n_probe,
        n=args.n,
        top_k=args.top_k,
        merkled=bool(args.merkled),
        B=B,
        log_2K=log_2K,
        M=M,
        K=K,
        d=d,
    )


def _compute_num_gates(cfg: GateConfig) -> int:
    try:
        from zk_IVF_PQ.zk_IVF_PQ import py_set_based_gate
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import `py_set_based_gate` from `zk_IVF_PQ.zk_IVF_PQ`. "
            "Build/install the Python extension first (e.g. `maturin develop`)."
        ) from e

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute num_gates for one set-based IVF-PQ configuration."
    )
    parser.add_argument("--D", type=int, required=True, help="Vector dimension.")
    parser.add_argument(
        "--n-list", dest="n_list", type=int, required=True, help="Number of IVF clusters."
    )
    parser.add_argument(
        "--n-probe", dest="n_probe", type=int, required=True, help="Number of probed clusters."
    )
    parser.add_argument(
        "--n", type=int, required=True, help="Number of points per probed cluster."
    )
    parser.add_argument(
        "--top-k", dest="top_k", type=int, default=64, help="Top-k used in the gadget."
    )
    parser.add_argument(
        "--merkled",
        action="store_true",
        help="Enable Merkle commitments in the gate count.",
    )

    parser.add_argument(
        "--B",
        type=int,
        default=None,
        help="Code budget B = M * log2(K). Use together with --log2K.",
    )
    parser.add_argument(
        "--log2K",
        "--log2-k",
        dest="log2_k",
        type=int,
        default=None,
        help="log2(K). Use together with --B.",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=None,
        help="Number of sub-vectors. Use together with --K.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=None,
        help="Codebook size per sub-vector. Use together with --M.",
    )

    args = parser.parse_args()
    cfg = _build_config(args)
    num_gates = _compute_num_gates(cfg)

    print("Configuration:")
    print(
        f"  D={cfg.D}, n_list={cfg.n_list}, n_probe={cfg.n_probe}, "
        f"n={cfg.n}, top_k={cfg.top_k}, merkled={int(cfg.merkled)}"
    )
    print(
        f"  B={cfg.B}, log2K={cfg.log_2K}, M={cfg.M}, K={cfg.K}, d={cfg.d}"
    )
    print(f"num_gates={num_gates}")


if __name__ == "__main__":
    main()
