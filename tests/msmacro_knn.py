from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

# Allow running as `python tests/msmacro_knn.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import faiss  # noqa: E402

from vec_data_load.ms_macro_load import (  # noqa: E402
    _read_qrels_unique_pairs,
    load_msmarco_collection_dev_unique_qrels,
)


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _load_json_or_none(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _meta_matches(meta: dict, *, expected: dict) -> bool:
    for key, value in expected.items():
        if meta.get(key) != value:
            return False
    return True


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _maybe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _ensure_float32_contiguous(x: np.ndarray, *, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={x.shape}")
    if not x.flags.c_contiguous:
        x = np.ascontiguousarray(x)
    return x


def _print_summary_from_rank(rank: np.ndarray, *, top_k: int) -> None:
    rank = np.asarray(rank)
    if rank.ndim != 1:
        raise ValueError(f"rank must be 1D, got shape={rank.shape}")
    if (rank == -1).any():
        remaining = int((rank == -1).sum())
        raise ValueError(f"rank contains {remaining} uncomputed entries (-1)")

    ks = [1, 5, 10, 20, 50, 100, 200, 500, min(1000, top_k), top_k]
    ks = [k for i, k in enumerate(ks) if k not in ks[:i]]

    print("\n[msmacro_knn] summary (hit/mrr/ndcg; single relevant; truncated):")
    for k in ks:
        in_k = (rank != 0) & (rank <= k)
        hit = float(in_k.mean())

        rr = np.zeros(rank.shape[0], dtype=np.float64)
        rr[in_k] = 1.0 / rank[in_k].astype(np.float64)
        mrr = float(rr.mean())

        nd = np.zeros(rank.shape[0], dtype=np.float64)
        nd[in_k] = 1.0 / np.log2(rank[in_k].astype(np.float64) + 1.0)
        ndcg = float(nd.mean())

        print(f"  k={k:4d}  hit={hit:.6f}  mrr={mrr:.6f}  ndcg={ndcg:.6f}")


def _report_if_cached_complete(
    *,
    out_dir: Path,
    top_k: int,
    expected_meta: dict,
) -> bool:
    meta_path = out_dir / f"meta_top{top_k}.json"
    qids_path = out_dir / "qids.npy"
    gt_pid_path = out_dir / "gt_pid.npy"
    gt_rank_path = out_dir / f"gt_rank_top{top_k}.npy"
    topk_ids_path = out_dir / f"topk_ids_top{top_k}.npy"
    topk_sims_path = out_dir / f"topk_sims_top{top_k}.npy"

    if not (
        meta_path.exists()
        and qids_path.exists()
        and gt_pid_path.exists()
        and gt_rank_path.exists()
        and topk_ids_path.exists()
        and topk_sims_path.exists()
    ):
        return False

    meta = _load_json_or_none(meta_path)
    if meta is None or not _meta_matches(meta, expected=expected_meta):
        return False

    gt_rank = np.load(gt_rank_path, mmap_mode="r")
    remaining = int((gt_rank == -1).sum())
    if remaining != 0:
        print(
            f"[msmacro_knn] cache partial: remaining={remaining}/{gt_rank.shape[0]} "
            f"({out_dir})"
        )
        return False

    shapes = meta.get("shapes") if isinstance(meta.get("shapes"), dict) else None
    if shapes is not None:
        base_shape = shapes.get("base")
        query_shape = shapes.get("queries")
        print(f"[msmacro_knn] cache hit: base={base_shape}, queries={query_shape}, top_k={top_k}")
    else:
        print(f"[msmacro_knn] cache hit: top_k={top_k}")
    _print_summary_from_rank(gt_rank, top_k=top_k)
    return True


def _open_or_create_memmap(
    *,
    path: Path,
    dtype: np.dtype,
    shape: Tuple[int, ...],
    fill_value,
    force_recompute: bool,
) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    if force_recompute:
        _maybe_unlink(path)

    if path.exists():
        arr = np.load(path, mmap_mode="r+")
        if tuple(arr.shape) != tuple(shape) or np.dtype(arr.dtype) != np.dtype(dtype):
            raise ValueError(
                f"Existing file has mismatched shape/dtype: {path} "
                f"(got {arr.shape}, {arr.dtype}; expected {shape}, {dtype})"
            )
        return arr

    arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    arr[:] = fill_value
    if hasattr(arr, "flush"):
        arr.flush()
    return arr


def compute_exact_msmarco_knn_topk(
    *,
    out_dir: Path,
    top_k: int = 1000,
    batch_queries: int = 64,
    threads: Optional[int] = None,
    max_db: Optional[int] = None,
    max_queries: Optional[int] = None,
    force_recompute: bool = False,
    collection_db_path: Path = Path("data/msmacro/collection.duckdb"),
    queries_db_path: Path = Path("data/msmacro/queries.dev.duckdb"),
    qrels_path: Path = Path("data/msmacro/qrels.dev.tsv"),
    cache_dir: Path = Path("data/msmacro/cache"),
) -> Path:
    """
    Compute exact (kNN, not ANN) top-k cosine/IP neighbors for MS MARCO dev queries
    (restricted to qrels qids that have exactly 1 relevant pid).

    Saves:
      - qids.npy:      (Q,) int64
      - gt_pid.npy:    (Q,) int32
      - gt_rank_topK.npy: (Q,) int16, -1=uncomputed, 0=not in topK, 1..K=rank within topK
      - gt_score.npy:  (Q,) float32, cosine/IP score of (query, gt_pid)
      - topk_ids_topK.npy:  (Q, K) int32
      - topk_sims_topK.npy: (Q, K) float32, cosine/IP scores
      - meta_topK.json: metadata (paths, shapes, config)
    """
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if batch_queries <= 0:
        raise ValueError("--batch-queries must be > 0")

    if threads is not None:
        threads = int(threads)
        if threads <= 0:
            raise ValueError("--threads must be > 0")
        faiss.omp_set_num_threads(threads)
        os.environ["OMP_NUM_THREADS"] = str(threads)

    expected_meta = {
        "kind": "msmarco_exact_knn_flat_ip",
        "top_k": int(top_k),
        "max_db": int(max_db) if max_db is not None else None,
        "max_queries": int(max_queries) if max_queries is not None else None,
        "collection_db_path": str(Path(collection_db_path).expanduser().resolve()),
        "queries_db_path": str(Path(queries_db_path).expanduser().resolve()),
        "qrels_path": str(Path(qrels_path).expanduser().resolve()),
        "cache_dir": str(Path(cache_dir).expanduser().resolve()),
    }
    if not force_recompute and _report_if_cached_complete(
        out_dir=out_dir, top_k=top_k, expected_meta=expected_meta
    ):
        return out_dir

    # Load embeddings (already row-normalized by loader).
    a, b, c = load_msmarco_collection_dev_unique_qrels(
        collection_db_path=collection_db_path,
        queries_db_path=queries_db_path,
        qrels_path=qrels_path,
        cache_dir=cache_dir,
        mmap_mode="r",
    )

    qids, pids = _read_qrels_unique_pairs(Path(qrels_path))
    if not np.array_equal(b, pids):
        raise RuntimeError("Loader output b does not match qrels unique pids; aborting.")

    base = _ensure_float32_contiguous(a, name="base")
    queries = _ensure_float32_contiguous(c, name="queries")
    gt_pids = np.asarray(b, dtype=np.int64)

    if max_db is not None:
        max_db = int(max_db)
        if max_db <= 0:
            raise ValueError("--max-db must be > 0")
        base = base[:max_db]
        keep = gt_pids < max_db
        qids = qids[keep]
        queries = queries[keep]
        gt_pids = gt_pids[keep]

    if max_queries is not None:
        max_queries = int(max_queries)
        if max_queries <= 0:
            raise ValueError("--max-queries must be > 0")
        qids = qids[:max_queries]
        queries = queries[:max_queries]
        gt_pids = gt_pids[:max_queries]

    n, d = base.shape
    q = int(queries.shape[0])
    if queries.shape[1] != d:
        raise ValueError(f"Dim mismatch: base D={d}, queries D={queries.shape[1]}")
    if top_k > n:
        raise ValueError(f"--top-k ({top_k}) must be <= database size ({n})")

    # Output paths
    meta_path = out_dir / f"meta_top{top_k}.json"
    qids_path = out_dir / "qids.npy"
    gt_pid_path = out_dir / "gt_pid.npy"
    gt_rank_path = out_dir / f"gt_rank_top{top_k}.npy"
    gt_score_path = out_dir / "gt_score.npy"
    topk_ids_path = out_dir / f"topk_ids_top{top_k}.npy"
    topk_sims_path = out_dir / f"topk_sims_top{top_k}.npy"

    if force_recompute:
        for p in (
            meta_path,
            qids_path,
            gt_pid_path,
            gt_rank_path,
            gt_score_path,
            topk_ids_path,
            topk_sims_path,
        ):
            _maybe_unlink(p)

    # Save stable small arrays.
    if qids_path.exists() and not force_recompute:
        on_disk_qids = np.load(qids_path)
        if not np.array_equal(on_disk_qids, qids):
            raise ValueError(
                f"{qids_path} exists but differs from current qids. "
                "Use --force-recompute or choose a different --out-dir."
            )
    else:
        np.save(qids_path, qids.astype(np.int64, copy=False))

    if gt_pid_path.exists() and not force_recompute:
        on_disk_gt = np.load(gt_pid_path)
        if not np.array_equal(on_disk_gt.astype(np.int64, copy=False), gt_pids):
            raise ValueError(
                f"{gt_pid_path} exists but differs from current gt pids. "
                "Use --force-recompute or choose a different --out-dir."
            )
    else:
        np.save(gt_pid_path, gt_pids.astype(np.int32, copy=False))

    # Create/resume large memmaps.
    gt_rank = _open_or_create_memmap(
        path=gt_rank_path,
        dtype=np.int16,
        shape=(q,),
        fill_value=np.int16(-1),
        force_recompute=force_recompute,
    )
    gt_score = _open_or_create_memmap(
        path=gt_score_path,
        dtype=np.float32,
        shape=(q,),
        fill_value=np.float32(np.nan),
        force_recompute=force_recompute,
    )
    topk_ids = _open_or_create_memmap(
        path=topk_ids_path,
        dtype=np.int32,
        shape=(q, top_k),
        fill_value=np.int32(-1),
        force_recompute=force_recompute,
    )
    topk_sims = _open_or_create_memmap(
        path=topk_sims_path,
        dtype=np.float32,
        shape=(q, top_k),
        fill_value=np.float32(np.nan),
        force_recompute=force_recompute,
    )

    # Write metadata early (so we can inspect settings while it's running).
    meta = {
        "created_at": _now_tag(),
        "kind": "msmarco_exact_knn_flat_ip",
        "top_k": int(top_k),
        "batch_queries": int(batch_queries),
        "threads": int(threads) if threads is not None else None,
        "max_db": int(max_db) if max_db is not None else None,
        "max_queries": int(max_queries) if max_queries is not None else None,
        "collection_db_path": str(Path(collection_db_path).expanduser().resolve()),
        "queries_db_path": str(Path(queries_db_path).expanduser().resolve()),
        "qrels_path": str(Path(qrels_path).expanduser().resolve()),
        "cache_dir": str(Path(cache_dir).expanduser().resolve()),
        "shapes": {"base": [int(n), int(d)], "queries": [int(q), int(d)]},
        "dtypes": {
            "base": str(base.dtype),
            "queries": str(queries.dtype),
            "topk_ids": str(topk_ids.dtype),
            "topk_sims": str(topk_sims.dtype),
            "gt_rank": str(gt_rank.dtype),
        },
    }
    _atomic_write_json(meta_path, meta)

    # Search and save.
    remaining = int((gt_rank == -1).sum())
    if remaining == 0:
        print("[msmacro_knn] cache hit (all queries already computed); skipping index build")
        _print_summary_from_rank(gt_rank, top_k=top_k)
        return out_dir

    # Build exact index (kNN, not ANN).
    print(f"[msmacro_knn] building IndexFlatIP: N={n}, D={d}")
    t0 = time.time()
    index = faiss.IndexFlatIP(d)
    index.add(base)
    build_s = time.time() - t0
    print(f"[msmacro_knn] index built in {build_s:.3f}s")

    print(f"[msmacro_knn] compute exact top{top_k}: Q={q} (remaining={remaining})")
    pbar = tqdm(total=remaining, desc=f"exact_top{top_k}", unit="q")

    try:
        for start in range(0, q, batch_queries):
            end = min(q, start + batch_queries)
            status = gt_rank[start:end]
            if np.all(status != -1):
                continue
            local = np.where(status == -1)[0]
            idx = (local + start).astype(np.int64, copy=False)

            q_batch = queries[idx]
            sims, ids = index.search(q_batch, top_k)

            ids = ids.astype(np.int32, copy=False)
            sims = sims.astype(np.float32, copy=False)

            # Rank of gt pid within top-k (1..K), or 0 if missing.
            gt = gt_pids[idx].astype(np.int32, copy=False)
            matches = ids == gt[:, None]
            found = matches.any(axis=1)
            rank = matches.argmax(axis=1).astype(np.int16) + 1
            rank[~found] = 0

            # Ground-truth similarity (query · base[gt_pid]).
            gt_vecs = base[gt.astype(np.int64, copy=False)]
            gt_sim = np.einsum("ij,ij->i", q_batch, gt_vecs).astype(np.float32, copy=False)

            topk_ids[idx] = ids
            topk_sims[idx] = sims
            gt_rank[idx] = rank
            gt_score[idx] = gt_sim

            if hasattr(topk_ids, "flush"):
                topk_ids.flush()
                topk_sims.flush()
                gt_rank.flush()
                gt_score.flush()

            pbar.update(int(idx.size))
    finally:
        pbar.close()

    _print_summary_from_rank(gt_rank, top_k=top_k)

    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute exact (kNN) top-k for MS MARCO embeddings using FAISS IndexFlatIP."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/msmacro/exact_knn"))
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--batch-queries", type=int, default=64)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--max-db", type=int, default=None, help="Debug: use only first N database vectors.")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Debug: use only first Q queries after filtering (unique qrels).",
    )
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/msmacro/cache"))
    parser.add_argument("--collection-db", type=Path, default=Path("data/msmacro/collection.duckdb"))
    parser.add_argument("--queries-db", type=Path, default=Path("data/msmacro/queries.dev.duckdb"))
    parser.add_argument("--qrels", type=Path, default=Path("data/msmacro/qrels.dev.tsv"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    compute_exact_msmarco_knn_topk(
        out_dir=args.out_dir,
        top_k=int(args.top_k),
        batch_queries=int(args.batch_queries),
        threads=args.threads,
        max_db=args.max_db,
        max_queries=args.max_queries,
        force_recompute=bool(args.force_recompute),
        collection_db_path=args.collection_db,
        queries_db_path=args.queries_db,
        qrels_path=args.qrels,
        cache_dir=args.cache_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
