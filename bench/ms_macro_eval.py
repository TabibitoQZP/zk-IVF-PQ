from __future__ import annotations

import sys
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

# Allow running as `python bench/ms_macro_eval.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ivf_pq import MAX_SCALE, rescale_query
from ivf_pq.standard import ivf_pq_learn as standard_ivf_pq_learn
from ivf_pq.standard import ivf_pq_query as standard_ivf_pq_query
from vec_data_load.ms_macro_load import (
    _read_qrels_unique_pairs,
    load_msmarco_collection_dev_unique_qrels,
)


@dataclass(frozen=True)
class MsMarcoEvalConfig:
    top_k: int
    num_runs: int

    n_list: int
    n_probe: int
    M: int
    K: int

    run_standard: bool
    run_zk: bool
    scale_n: int
    cluster_bound: Optional[int]
    seed: int

    max_db: Optional[int]
    max_queries: Optional[int]


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _normalize_rows_inplace(x: np.ndarray, *, eps: float = 1e-12) -> None:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    x /= norms


def _rescale_database_to_memmap(
    base: np.ndarray,
    *,
    out_path: Path,
    scale_n: int,
    chunk_rows: int = 4096,
    dtype: np.dtype = np.int32,
) -> Tuple[np.ndarray, float, float]:
    """
    Streamingly rescale float base vectors into integer range [0, scale_n),
    writing the result to a .npy memmap file to avoid huge peak memory.
    """
    if scale_n <= 1:
        raise ValueError("scale_n must be > 1")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be > 0")

    base = np.asarray(base)
    if base.ndim != 2:
        raise ValueError("base must be a 2D array")
    n, d = base.shape

    v_min = float("inf")
    v_max = float("-inf")
    for start in tqdm(range(0, n, chunk_rows), desc="scan min/max", unit="row"):
        block = base[start : start + chunk_rows]
        if block.size == 0:
            continue
        v_min = min(v_min, float(block.min()))
        v_max = max(v_max, float(block.max()))

    if not np.isfinite(v_min) or not np.isfinite(v_max):
        raise ValueError("Failed to compute v_min/v_max for rescaling")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    scaled = np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype, shape=(n, d))

    if v_max == v_min:
        scaled.fill(0)
        scaled.flush()
        return scaled, v_min, v_max

    scale = (scale_n - 1) / (v_max - v_min)
    for start in tqdm(range(0, n, chunk_rows), desc="rescale base", unit="row"):
        block = base[start : start + chunk_rows].astype(np.float64, copy=False)
        scaled_block = np.rint((block - v_min) * scale).astype(dtype, copy=False)
        np.clip(scaled_block, 0, scale_n - 1, out=scaled_block)
        scaled[start : start + scaled_block.shape[0]] = scaled_block

    scaled.flush()
    return scaled, v_min, v_max


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_msmarco_eval(
    *,
    out_dir: Path,
    config: MsMarcoEvalConfig,
    collection_db_path: Path,
    queries_db_path: Path,
    qrels_path: Path,
    cache_dir: Path,
    force_recompute: bool,
) -> Path:
    """
    For each run:
      1) Train IVF-PQ (standard / zk) from scratch.
      2) Query all selected dev queries and collect top-k doc ids.
      3) Save (qids, gt_pid, topk) to out_dir for later analysis.
    """
    out_dir = out_dir.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load (a,b,c) from the pre-built DuckDB caches.
    a, b, c = load_msmarco_collection_dev_unique_qrels(
        collection_db_path=collection_db_path,
        queries_db_path=queries_db_path,
        qrels_path=qrels_path,
        cache_dir=cache_dir,
        mmap_mode="r",
    )

    qids, pids = _read_qrels_unique_pairs(qrels_path)
    if b.shape != pids.shape or not np.array_equal(b, pids):
        raise ValueError("Internal mismatch: load() output b does not match qrels unique pids")

    base = a
    queries = c
    gt_pids = b

    if config.max_db is not None:
        max_db = int(config.max_db)
        if max_db <= 0:
            raise ValueError("--max-db must be > 0")
        base = base[:max_db]
        keep = gt_pids < max_db
        qids = qids[keep]
        queries = queries[keep]
        gt_pids = gt_pids[keep]

    if config.max_queries is not None:
        max_q = int(config.max_queries)
        if max_q <= 0:
            raise ValueError("--max-queries must be > 0")
        qids = qids[:max_q]
        queries = queries[:max_q]
        gt_pids = gt_pids[:max_q]

    if config.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if config.top_k > base.shape[0]:
        raise ValueError(f"--top-k ({config.top_k}) must be <= database size ({base.shape[0]})")

    q = int(queries.shape[0])
    n, d = base.shape
    print(f"[msmacro_eval] db={n}x{d}, queries={q}x{queries.shape[1]}")

    # Ensure float32 and normalized (should already be normalized by loader).
    base = np.asarray(base, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    _normalize_rows_inplace(queries)

    # Optional: prepare scaled base for zk once, reused across runs.
    scaled_base = None
    v_min = None
    v_max = None
    scaled_meta_path = out_dir / "scaled_base_meta.json"
    scaled_base_path = (
        out_dir
        / f"scaled_base_N{n}_D{d}_scale{config.scale_n}_dtypei32.npy"
    )

    if config.run_zk:
        meta = None
        if not force_recompute and scaled_meta_path.exists() and scaled_base_path.exists():
            try:
                meta = json.loads(scaled_meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = None

        expected_meta = {
            "collection_db": str(collection_db_path.expanduser().resolve()),
            "queries_db": str(queries_db_path.expanduser().resolve()),
            "qrels": str(qrels_path.expanduser().resolve()),
            "scale_n": int(config.scale_n),
            "shape": [int(n), int(d)],
        }

        if meta is not None and meta.get("inputs") == expected_meta:
            scaled_base = np.load(scaled_base_path, mmap_mode="r")
            v_min = float(meta["v_min"])
            v_max = float(meta["v_max"])
            print(f"[msmacro_eval] loaded scaled_base cache: {scaled_base_path}")
        else:
            scaled_base, v_min, v_max = _rescale_database_to_memmap(
                base,
                out_path=scaled_base_path,
                scale_n=config.scale_n,
                chunk_rows=4096,
                dtype=np.int32,
            )
            _save_json(
                scaled_meta_path,
                {"inputs": expected_meta, "v_min": float(v_min), "v_max": float(v_max)},
            )
            scaled_base = np.load(scaled_base_path, mmap_mode="r")

    results_dir = out_dir / "runs"
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg_payload = {
        "created_at": _now_tag(),
        "collection_db_path": str(collection_db_path),
        "queries_db_path": str(queries_db_path),
        "qrels_path": str(qrels_path),
        "effective_db_shape": [int(n), int(d)],
        "effective_queries": int(qids.shape[0]),
        "config": config.__dict__,
    }
    _save_json(out_dir / "config.json", cfg_payload)

    for run_idx in range(config.num_runs):
        run_out = results_dir / f"run_{run_idx:03d}.npz"
        run_meta_out = results_dir / f"run_{run_idx:03d}.json"
        if run_out.exists() and run_meta_out.exists() and not force_recompute:
            print(f"[msmacro_eval] skip existing: {run_out.name}")
            continue

        run_seed_std = int(config.seed + run_idx)
        run_seed_zk = int(config.seed + 10_000 + run_idx)

        run_meta = {
            "run_idx": int(run_idx),
            "top_k": int(config.top_k),
            "n_list": int(config.n_list),
            "n_probe": int(config.n_probe),
            "M": int(config.M),
            "K": int(config.K),
            "run_standard": bool(config.run_standard),
            "run_zk": bool(config.run_zk),
            "scale_n": int(config.scale_n),
            "cluster_bound": config.cluster_bound,
            "seed_standard": run_seed_std,
            "seed_zk": run_seed_zk,
            "v_min": float(v_min) if v_min is not None else None,
            "v_max": float(v_max) if v_max is not None else None,
        }

        payload = {
            "qids": qids.astype(np.int64, copy=False),
            "gt_pid": gt_pids.astype(np.int64, copy=False),
        }

        if config.run_standard:
            t0 = time.time()
            std_labels, std_center, std_code_books, std_quant_vecs, std_id_groups = (
                standard_ivf_pq_learn(
                    base,
                    n_list=config.n_list,
                    M=config.M,
                    K=config.K,
                    random_state=run_seed_std,
                )
            )
            run_meta["standard_train_time_s"] = float(time.time() - t0)

            std_topk = np.full((qids.shape[0], config.top_k), -1, dtype=np.int64)
            t0 = time.time()
            for i in tqdm(range(qids.shape[0]), desc=f"run{run_idx:03d} standard", unit="q"):
                pred = standard_ivf_pq_query(
                    queries[i],
                    config.top_k,
                    std_labels,
                    std_center,
                    std_code_books,
                    std_quant_vecs,
                    std_id_groups,
                    n_probe=config.n_probe,
                )
                pred = np.asarray(pred, dtype=np.int64)
                n_take = min(config.top_k, int(pred.shape[0]))
                if n_take > 0:
                    std_topk[i, :n_take] = pred[:n_take]
            run_meta["standard_query_time_s"] = float(time.time() - t0)
            payload["standard_topk"] = std_topk

        if config.run_zk:
            try:
                from ivf_pq.merkle_zk import ivf_pq_learn as zk_ivf_pq_learn
                from ivf_pq.merkle_zk import zk_ivf_pq_query
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "Failed to import ZK IVF-PQ Python bindings. "
                    "Build/install the extension first (e.g. `maturin develop`)."
                ) from e

            assert scaled_base is not None and v_min is not None and v_max is not None

            t0 = time.time()
            if config.cluster_bound is not None:
                (
                    zk_labels,
                    zk_center,
                    zk_code_books,
                    zk_quant_vecs,
                    zk_id_groups,
                    zk_changed_count,
                ) = zk_ivf_pq_learn(
                    scaled_base,
                    n_list=config.n_list,
                    M=config.M,
                    K=config.K,
                    random_state=run_seed_zk,
                    cluster_bound=int(config.cluster_bound),
                )
                run_meta["zk_changed_count"] = int(zk_changed_count)
            else:
                (
                    zk_labels,
                    zk_center,
                    zk_code_books,
                    zk_quant_vecs,
                    zk_id_groups,
                ) = zk_ivf_pq_learn(
                    scaled_base,
                    n_list=config.n_list,
                    M=config.M,
                    K=config.K,
                    random_state=run_seed_zk,
                )
            run_meta["zk_train_time_s"] = float(time.time() - t0)

            zk_topk = np.full((qids.shape[0], config.top_k), -1, dtype=np.int64)
            t0 = time.time()
            for i in tqdm(range(qids.shape[0]), desc=f"run{run_idx:03d} zk", unit="q"):
                scaled_query = rescale_query(queries[i], config.scale_n, v_min, v_max)
                pred, _ = zk_ivf_pq_query(
                    scaled_query,
                    zk_center,
                    zk_code_books,
                    zk_quant_vecs,
                    zk_id_groups,
                    top_k=config.top_k,
                    n_probe=config.n_probe,
                    proof=False,
                )
                pred = np.asarray(pred, dtype=np.int64)
                n_take = min(config.top_k, int(pred.shape[0]))
                if n_take > 0:
                    zk_topk[i, :n_take] = pred[:n_take]
            run_meta["zk_query_time_s"] = float(time.time() - t0)
            payload["zk_topk"] = zk_topk

        _save_npz(run_out, **payload)
        _save_json(run_meta_out, run_meta)
        print(f"[msmacro_eval] saved: {run_out}")

    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MS MARCO dev eval runner: train IVF-PQ multiple times and save top-k predictions for analysis."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/msmacro/ms_macro_eval"))
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=100)

    parser.add_argument("--n-list", type=int, default=8192)
    parser.add_argument("--n-probe", type=int, default=64)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--K", type=int, default=256)

    parser.add_argument(
        "--standard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run standard IVF-PQ (default: enabled).",
    )
    parser.add_argument(
        "--zk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ZK IVF-PQ (default: enabled).",
    )
    parser.add_argument("--scale-n", type=int, default=MAX_SCALE)
    parser.add_argument("--cluster-bound", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max-db", type=int, default=None, help="Use only first N database vectors (debug).")
    parser.add_argument(
        "--max-queries", type=int, default=None, help="Use only first Q queries after filtering (debug)."
    )

    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/msmacro/cache"))
    parser.add_argument("--collection-db", type=Path, default=Path("data/msmacro/collection.duckdb"))
    parser.add_argument("--queries-db", type=Path, default=Path("data/msmacro/queries.dev.duckdb"))
    parser.add_argument("--qrels", type=Path, default=Path("data/msmacro/qrels.dev.tsv"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg = MsMarcoEvalConfig(
        top_k=int(args.top_k),
        num_runs=int(args.num_runs),
        n_list=int(args.n_list),
        n_probe=int(args.n_probe),
        M=int(args.M),
        K=int(args.K),
        run_standard=bool(args.standard),
        run_zk=bool(args.zk),
        scale_n=int(args.scale_n),
        cluster_bound=args.cluster_bound,
        seed=int(args.seed),
        max_db=args.max_db,
        max_queries=args.max_queries,
    )

    if not cfg.run_standard and not cfg.run_zk:
        raise SystemExit("At least one of --standard/--zk must be enabled.")

    run_msmarco_eval(
        out_dir=args.out_dir,
        config=cfg,
        collection_db_path=args.collection_db,
        queries_db_path=args.queries_db,
        qrels_path=args.qrels,
        cache_dir=args.cache_dir,
        force_recompute=bool(args.force_recompute),
    )


if __name__ == "__main__":
    main()
