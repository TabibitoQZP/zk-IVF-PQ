from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Allow running as `python bench/ms_macro_result.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _fingerprint(path: Path) -> Dict[str, object]:
    st = path.stat()
    return {"path": str(path.resolve()), "mtime_ns": int(st.st_mtime_ns), "size": int(st.st_size)}


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(payload).__name__}")
    return payload


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _compute_rank_1based(
    topk_ids: np.ndarray,
    gt_pid: np.ndarray,
    *,
    block_rows: int = 4096,
) -> np.ndarray:
    """
    Returns: (Q,) int16 with 0=miss, 1..K=rank (1-based).
    """
    topk_ids = np.asarray(topk_ids)
    gt_pid = np.asarray(gt_pid)
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be 2D, got shape={topk_ids.shape}")
    if gt_pid.ndim != 1:
        raise ValueError(f"gt_pid must be 1D, got shape={gt_pid.shape}")
    if topk_ids.shape[0] != gt_pid.shape[0]:
        raise ValueError(
            f"Q mismatch: topk_ids Q={topk_ids.shape[0]} vs gt_pid Q={gt_pid.shape[0]}"
        )
    if block_rows <= 0:
        raise ValueError("block_rows must be > 0")

    q, k = topk_ids.shape
    out_dtype = np.int16 if k <= np.iinfo(np.int16).max else np.int32
    out = np.zeros((q,), dtype=out_dtype)

    for start in range(0, q, block_rows):
        end = min(q, start + block_rows)
        topk_block = topk_ids[start:end]
        gt_block = gt_pid[start:end]

        match = topk_block == gt_block[:, None]
        any_match = match.any(axis=1)
        if any_match.any():
            rows = np.nonzero(any_match)[0]
            first = match.argmax(axis=1)[rows].astype(out_dtype) + 1
            out[start + rows] = first
    return out


def _metrics_from_rank(
    rank_1based: np.ndarray, *, ks: Sequence[int]
) -> Dict[int, Dict[str, float]]:
    rank = np.asarray(rank_1based)
    if rank.ndim != 1:
        raise ValueError(f"rank must be 1D, got shape={rank.shape}")
    if (rank < 0).any():
        raise ValueError("rank contains negative value(s)")

    out: Dict[int, Dict[str, float]] = {}
    rank_f = rank.astype(np.float64, copy=False)
    for k in ks:
        k = int(k)
        in_k = (rank != 0) & (rank <= k)
        hit = float(in_k.mean())

        rr = np.zeros(rank.shape[0], dtype=np.float64)
        rr[in_k] = 1.0 / rank_f[in_k]
        mrr = float(rr.mean())

        nd = np.zeros(rank.shape[0], dtype=np.float64)
        nd[in_k] = 1.0 / np.log2(rank_f[in_k] + 1.0)
        ndcg = float(nd.mean())

        out[k] = {"hit": hit, "mrr": mrr, "ndcg": ndcg}
    return out


def _mean_ci95(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "n": 0}
    mean = float(arr.mean())
    if arr.size > 1:
        std = float(arr.std(ddof=1))
        ci95 = float(1.96 * std / math.sqrt(int(arr.size)))
    else:
        ci95 = 0.0
    return {"mean": mean, "ci95": ci95, "n": int(arr.size)}


def _metrics_from_rank_with_query_ci(
    rank_1based: np.ndarray, *, ks: Sequence[int]
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Per-k metrics with 95% CI estimated across queries (not across runs).

    Returns:
      {k: {"hit": {"mean":..., "ci95_query":..., "n_queries":...}, ...}, ...}
    """
    rank = np.asarray(rank_1based)
    if rank.ndim != 1:
        raise ValueError(f"rank must be 1D, got shape={rank.shape}")
    if (rank < 0).any():
        raise ValueError("rank contains negative value(s)")

    q = int(rank.shape[0])
    if q <= 0:
        raise ValueError("rank must be non-empty")

    rank_f = rank.astype(np.float64, copy=False)
    denom = float(q)
    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    for k in ks:
        k = int(k)
        in_k = (rank != 0) & (rank <= k)
        n_hit = int(in_k.sum())
        hit_mean = float(n_hit / denom)
        if q > 1:
            # Bernoulli std (sample) via p(1-p)*n/(n-1).
            hit_var_pop = hit_mean * (1.0 - hit_mean)
            hit_std = math.sqrt(hit_var_pop * denom / (denom - 1.0))
            hit_ci = float(1.96 * hit_std / math.sqrt(denom))
        else:
            hit_ci = 0.0

        r = rank_f[in_k]
        if r.size:
            rr = 1.0 / r
            nd = 1.0 / np.log2(r + 1.0)

            mrr_mean = float(rr.sum() / denom)
            ndcg_mean = float(nd.sum() / denom)

            if q > 1:
                rr_mean2 = float((rr * rr).sum() / denom)
                nd_mean2 = float((nd * nd).sum() / denom)

                rr_var_pop = max(0.0, rr_mean2 - mrr_mean * mrr_mean)
                nd_var_pop = max(0.0, nd_mean2 - ndcg_mean * ndcg_mean)

                rr_std = math.sqrt(rr_var_pop * denom / (denom - 1.0))
                nd_std = math.sqrt(nd_var_pop * denom / (denom - 1.0))

                mrr_ci = float(1.96 * rr_std / math.sqrt(denom))
                ndcg_ci = float(1.96 * nd_std / math.sqrt(denom))
            else:
                mrr_ci = 0.0
                ndcg_ci = 0.0
        else:
            mrr_mean = 0.0
            ndcg_mean = 0.0
            mrr_ci = 0.0
            ndcg_ci = 0.0

        out[k] = {
            "hit": {"mean": hit_mean, "ci95_query": hit_ci, "n_queries": q},
            "mrr": {"mean": mrr_mean, "ci95_query": mrr_ci, "n_queries": q},
            "ndcg": {"mean": ndcg_mean, "ci95_query": ndcg_ci, "n_queries": q},
        }
    return out


def _run_idx_from_name(name: str) -> int:
    # run_000.npz -> 0
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 2 or parts[0] != "run":
        raise ValueError(f"Unexpected run filename: {name!r}")
    return int(parts[1])


def _expected_rank_cache_inputs(
    *,
    run_npz_path: Path,
    scheme: str,
    shape_topk: Tuple[int, int],
) -> Dict[str, object]:
    q, k = shape_topk
    return {
        "run_npz": _fingerprint(run_npz_path),
        "scheme": str(scheme),
        "shape_topk": [int(q), int(k)],
    }


def _rank_cache_paths(*, result_dir: Path, scheme: str, run_idx: int) -> Tuple[Path, Path]:
    rank_path = result_dir / f"rank_{scheme}_run_{run_idx:03d}.npy"
    meta_path = result_dir / f"rank_{scheme}_run_{run_idx:03d}.json"
    return rank_path, meta_path


def _maybe_load_rank_cache(
    *,
    result_dir: Path,
    scheme: str,
    run_idx: int,
    expected_inputs: Dict[str, object],
    q: int,
    force: bool,
) -> Optional[np.ndarray]:
    if force:
        return None
    rank_path, meta_path = _rank_cache_paths(result_dir=result_dir, scheme=scheme, run_idx=run_idx)
    if not rank_path.exists() or not meta_path.exists():
        return None

    try:
        meta = _load_json(meta_path)
    except Exception:
        return None

    if not (isinstance(meta, dict) and meta.get("inputs") == expected_inputs):
        return None

    try:
        rank = np.load(rank_path)
    except Exception:
        return None
    if not (isinstance(rank, np.ndarray) and rank.shape == (int(q),)):
        return None
    return rank


def _save_rank_cache(
    *,
    result_dir: Path,
    scheme: str,
    run_idx: int,
    expected_inputs: Dict[str, object],
    rank: np.ndarray,
) -> None:
    rank_path, meta_path = _rank_cache_paths(result_dir=result_dir, scheme=scheme, run_idx=run_idx)
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(rank_path, np.asarray(rank))
    _atomic_write_json(meta_path, {"created_at": _now_tag(), "inputs": expected_inputs})


def evaluate_msmarco_eval_dir(
    exp_dir: Path,
    *,
    ks: Sequence[int],
    force: bool,
    block_rows: int,
) -> Path:
    exp_dir = exp_dir.expanduser().resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")

    runs_dir = exp_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Missing runs dir: {runs_dir}")

    run_npzs = sorted(runs_dir.glob("run_*.npz"))
    if not run_npzs:
        raise FileNotFoundError(f"No run_*.npz found under: {runs_dir}")

    cfg_path = exp_dir / "config.json"
    cfg = _load_json(cfg_path) if cfg_path.exists() else None

    result_dir = exp_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    per_run: List[dict] = []
    all_means: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
    all_query_ci: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
    all_qps: Dict[str, List[float]] = {"standard": [], "zk": []}

    for run_npz_path in run_npzs:
        run_idx = _run_idx_from_name(run_npz_path.name)
        run_meta_path = runs_dir / f"run_{run_idx:03d}.json"
        run_meta = _load_json(run_meta_path) if run_meta_path.exists() else {}

        with np.load(run_npz_path) as data:
            qids = np.asarray(data["qids"], dtype=np.int64)
            gt_pid = np.asarray(data["gt_pid"], dtype=np.int64)
            q = int(qids.shape[0])

            run_entry: dict = {
                "run_idx": int(run_idx),
                "run_npz": str(run_npz_path),
                "Q": q,
                "meta": run_meta,
                "metrics": {},
            }

            cfg_inner = cfg.get("config") if isinstance(cfg, dict) else None
            default_topk = cfg_inner.get("top_k") if isinstance(cfg_inner, dict) else None

            for scheme, topk_key in (("standard", "standard_topk"), ("zk", "zk_topk")):
                if topk_key not in data.files:
                    continue

                run_topk = run_meta.get("top_k", default_topk)
                run_topk = int(run_topk) if run_topk is not None else None

                # Try cached ranks first to avoid loading huge top-k arrays.
                if run_topk is not None:
                    expected_inputs = _expected_rank_cache_inputs(
                        run_npz_path=run_npz_path,
                        scheme=scheme,
                        shape_topk=(q, run_topk),
                    )
                    rank = _maybe_load_rank_cache(
                        result_dir=result_dir,
                        scheme=scheme,
                        run_idx=run_idx,
                        expected_inputs=expected_inputs,
                        q=q,
                        force=force,
                    )
                else:
                    expected_inputs = None
                    rank = None

                # Cache miss -> load top-k ids and compute ranks.
                if rank is None:
                    topk_ids = np.asarray(data[topk_key], dtype=np.int64)
                    if topk_ids.shape[0] != q:
                        raise ValueError(
                            f"{run_npz_path} key={topk_key} has mismatched Q: {topk_ids.shape[0]} vs {q}"
                        )
                    run_topk = int(topk_ids.shape[1])
                    expected_inputs = _expected_rank_cache_inputs(
                        run_npz_path=run_npz_path,
                        scheme=scheme,
                        shape_topk=(q, run_topk),
                    )
                    rank = _compute_rank_1based(topk_ids, gt_pid, block_rows=block_rows)
                    _save_rank_cache(
                        result_dir=result_dir,
                        scheme=scheme,
                        run_idx=run_idx,
                        expected_inputs=expected_inputs,
                        rank=rank,
                    )

                    # Help GC (topk arrays can be huge).
                    del topk_ids

                # Clamp ks to available top_k for this run.
                assert run_topk is not None
                ks_eff = [int(k) for k in ks if int(k) <= int(run_topk)]
                if not ks_eff:
                    raise ValueError(f"No ks <= top_k={run_topk} for {run_npz_path}")

                metrics_ci = _metrics_from_rank_with_query_ci(rank, ks=ks_eff)
                run_entry["metrics"][scheme] = {"ks": ks_eff, "metrics": metrics_ci}

                # Aggregate.
                scheme_bucket = all_means.setdefault(scheme, {})
                scheme_qci_bucket = all_query_ci.setdefault(scheme, {})
                for k, m in metrics_ci.items():
                    k_bucket = scheme_bucket.setdefault(int(k), {})
                    k_qci_bucket = scheme_qci_bucket.setdefault(int(k), {})
                    for metric_name, record in m.items():
                        k_bucket.setdefault(metric_name, []).append(float(record["mean"]))
                        k_qci_bucket.setdefault(metric_name, []).append(float(record["ci95_query"]))

                # Timing / QPS (if present).
                qt = run_meta.get(f"{scheme}_query_time_s")
                if isinstance(qt, (int, float)) and float(qt) > 0:
                    all_qps[scheme].append(float(q / float(qt)))

            per_run.append(run_entry)

    # Build aggregate summary with mean + CI95 across runs.
    aggregate: dict = {"metrics": {}, "qps": {}}
    for scheme, scheme_bucket in all_means.items():
        scheme_out: dict = {}
        for k, k_bucket in scheme_bucket.items():
            k_out: dict = {}
            for metric_name, values in k_bucket.items():
                run_stat = _mean_ci95(values)
                qci_values = (
                    all_query_ci.get(scheme, {}).get(int(k), {}).get(metric_name, [])
                )
                qci_mean = float(np.mean(qci_values)) if qci_values else float("nan")
                k_out[metric_name] = {
                    "mean": float(run_stat["mean"]),
                    "ci95_runs": float(run_stat["ci95"]),
                    "n_runs": int(run_stat["n"]),
                    "ci95_query": qci_mean,
                }
            scheme_out[str(int(k))] = k_out
        aggregate["metrics"][scheme] = scheme_out

    for scheme, qps_list in all_qps.items():
        if qps_list:
            aggregate["qps"][scheme] = _mean_ci95(qps_list)

    report = {
        "created_at": _now_tag(),
        "exp_dir": str(exp_dir),
        "config": cfg,
        "ks": [int(k) for k in ks],
        "runs": per_run,
        "aggregate": aggregate,
    }

    report_path = result_dir / "report.json"
    _atomic_write_json(report_path, report)

    # Human-readable summary.
    lines: List[str] = []
    lines.append(f"[ms_macro_result] exp_dir={exp_dir}")
    if isinstance(cfg, dict):
        eff = cfg.get("effective_queries")
        db_shape = cfg.get("effective_db_shape")
        lines.append(f"  effective_db_shape={db_shape} effective_queries={eff}")
        cfg_inner = cfg.get("config") if isinstance(cfg.get("config"), dict) else None
        if cfg_inner is not None:
            lines.append(
                "  cfg: "
                + " ".join(
                    f"{k}={cfg_inner.get(k)}"
                    for k in ("top_k", "num_runs", "n_list", "n_probe", "M", "K", "cluster_bound", "scale_n")
                    if k in cfg_inner
                )
            )
    lines.append("")
    for scheme in ("standard", "zk"):
        if scheme not in report["aggregate"]["metrics"]:
            continue
        lines.append(f"{scheme}:")
        if scheme in report["aggregate"].get("qps", {}):
            qps = report["aggregate"]["qps"][scheme]
            lines.append(f"  qps mean={qps['mean']:.3f} ci95={qps['ci95']:.3f} (n={qps['n']})")
        metrics_s = report["aggregate"]["metrics"][scheme]

        # CI policy: if only one run, report query-level CI; otherwise report run-level CI.
        n_runs = None
        for _k_str, _k_rec in metrics_s.items():
            if isinstance(_k_rec, dict) and isinstance(_k_rec.get("hit"), dict) and "n_runs" in _k_rec["hit"]:
                n_runs = int(_k_rec["hit"]["n_runs"])
                break
        if n_runs is None:
            n_runs = 1
        if n_runs <= 1:
            n_queries = int(report.get("runs", [{}])[0].get("Q", 0)) if isinstance(report.get("runs"), list) else 0
            if n_queries > 0:
                lines.append(f"  ci95: across queries (n_queries={n_queries})")
        else:
            lines.append(f"  ci95: across runs (n_runs={n_runs})")

        for k in sorted(int(x) for x in metrics_s.keys()):
            m = metrics_s[str(k)]
            hit = m.get("hit", {})
            mrr = m.get("mrr", {})
            ndcg = m.get("ndcg", {})
            if hit and mrr and ndcg:
                hit_ci = float(hit["ci95_query"] if n_runs <= 1 else hit["ci95_runs"])
                mrr_ci = float(mrr["ci95_query"] if n_runs <= 1 else mrr["ci95_runs"])
                ndcg_ci = float(ndcg["ci95_query"] if n_runs <= 1 else ndcg["ci95_runs"])
                lines.append(
                    f"  k={k:4d}  "
                    f"hit={hit['mean']:.6f}±{hit_ci:.6f}  "
                    f"mrr={mrr['mean']:.6f}±{mrr_ci:.6f}  "
                    f"ndcg={ndcg['mean']:.6f}±{ndcg_ci:.6f}"
                )
        lines.append("")

    summary_path = result_dir / "summary.txt"
    _atomic_write_text(summary_path, "\n".join(lines).rstrip() + "\n")

    print("\n".join(lines))
    print(f"[ms_macro_result] saved: {report_path}")
    print(f"[ms_macro_result] saved: {summary_path}")
    return report_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate outputs of `bench.ms_macro_eval` runs: compute hit/mrr/ndcg at multiple k "
            "from saved (gt_pid, topk_ids), and write a report under EXP_DIR/result/."
        )
    )
    parser.add_argument(
        "exp_dir",
        type=Path,
        nargs="+",
        help="One or more experiment dirs produced by `bench.ms_macro_eval` (contains runs/run_*.npz).",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="1,5,10,20,50,100,200,500,1000",
        help="Comma-separated k list for hit/mrr/ndcg (default: 1,5,10,20,50,100,200,500,1000).",
    )
    parser.add_argument("--force", action="store_true", help="Recompute cached ranks and overwrite reports.")
    parser.add_argument(
        "--block-rows",
        type=int,
        default=4096,
        help="Row block size when computing ranks (controls memory).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ks = _parse_int_list(str(args.ks))
    if not ks:
        raise SystemExit("--ks must be a non-empty comma-separated list")
    if any(int(k) <= 0 for k in ks):
        raise SystemExit("--ks values must be positive")

    for exp_dir in args.exp_dir:
        evaluate_msmarco_eval_dir(
            exp_dir,
            ks=ks,
            force=bool(args.force),
            block_rows=int(args.block_rows),
        )


if __name__ == "__main__":
    main()
