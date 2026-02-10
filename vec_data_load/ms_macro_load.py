from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import duckdb
import numpy as np


PathLike = Union[str, Path]


def _fingerprint(path: Path) -> Dict[str, object]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime_ns": int(st.st_mtime_ns),
        "size": int(st.st_size),
    }


def _read_qrels_unique_pairs(qrels_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse MS MARCO qrels TSV (format: qid, 0, pid, rel).

    Returns:
      qids: (k,) int64 sorted ascending
      pids: (k,) int64 aligned with qids

    Keeps only qids that have exactly 1 relevant pid (rel > 0).
    """
    qrels_path = qrels_path.expanduser().resolve()
    if not qrels_path.exists():
        raise FileNotFoundError(f"qrels file not found: {qrels_path}")

    pid_by_qid: Dict[int, Optional[int]] = {}
    cnt_by_qid: Dict[int, int] = {}

    with qrels_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                parts = line.split()
            if len(parts) < 4:
                raise ValueError(
                    f"Malformed qrels line {line_no}: expected 4 columns (qid, 0, pid, rel), got: {raw!r}"
                )

            try:
                qid = int(parts[0])
                pid = int(parts[2])
                rel = int(parts[3])
            except ValueError as e:
                raise ValueError(f"Malformed qrels line {line_no}: {raw!r}") from e

            if rel <= 0:
                continue

            if qid not in cnt_by_qid:
                cnt_by_qid[qid] = 1
                pid_by_qid[qid] = pid
            else:
                cnt_by_qid[qid] += 1
                pid_by_qid[qid] = None

    qids = [qid for qid, c in cnt_by_qid.items() if c == 1 and pid_by_qid.get(qid) is not None]
    qids.sort()
    pids = [int(pid_by_qid[qid]) for qid in qids]

    return np.asarray(qids, dtype=np.int64), np.asarray(pids, dtype=np.int64)


def _fixed_size_list_to_numpy_2d(arr, dim: int, dtype: np.dtype) -> np.ndarray:
    # arr: pyarrow.FixedSizeListArray (or ChunkedArray/Array of it)
    # We avoid importing pyarrow directly; duckdb provides these objects.
    if hasattr(arr, "chunk") and hasattr(arr, "num_chunks"):
        chunks = [arr.chunk(i) for i in range(arr.num_chunks)]
        out_chunks = [_fixed_size_list_to_numpy_2d(chunk, dim, dtype) for chunk in chunks]
        return np.concatenate(out_chunks, axis=0) if len(out_chunks) > 1 else out_chunks[0]

    flat = arr.values.to_numpy(zero_copy_only=False)
    out = flat.reshape((len(arr), dim))
    if out.dtype != dtype:
        out = out.astype(dtype, copy=False)
    return out


def _normalize_rows_inplace(x: np.ndarray, *, eps: float = 1e-12, chunk_rows: int = 65536) -> None:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be > 0")

    n = x.shape[0]
    for start in range(0, n, chunk_rows):
        block = x[start : start + chunk_rows]
        norms = np.sqrt(np.einsum("ij,ij->i", block, block))
        norms = np.maximum(norms, eps)
        block /= norms[:, None]


def _ensure_cache_dir(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)


def _load_meta(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _meta_matches(meta: dict, *, expected: dict) -> bool:
    for key, value in expected.items():
        if meta.get(key) != value:
            return False
    return True


def load_msmarco_collection_dev_unique_qrels(
    *,
    collection_db_path: PathLike = "data/msmacro/collection.duckdb",
    collection_table_name: str = "collection_embeddings",
    queries_db_path: PathLike = "data/msmacro/queries.dev.duckdb",
    queries_table_name: str = "queries_dev_embeddings",
    qrels_path: PathLike = "data/msmacro/qrels.dev.tsv",
    cache_dir: PathLike = "data/msmacro/cache",
    dtype: np.dtype = np.float32,
    use_cache: bool = True,
    force_recompute: bool = False,
    mmap_mode: Optional[str] = "r",
    record_batch_size: int = 65536,
    normalize_chunk_rows: int = 65536,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-computed MS MARCO embeddings from DuckDB and return (a, b, c).

    - a: (N, D) collection embeddings where a[pid] corresponds to passage id (pid).
    - b: (k,) relevant pids from qrels.dev where each selected qid has exactly 1 relevant pid.
    - c: (k, D) query embeddings aligned with b (same ordering), all row-wise L2 normalized.

    Caching:
      If cache files exist and match input file fingerprints, loads from cache;
      otherwise reads from DuckDB/TSV and writes cache for next run.
    """
    collection_db_path = Path(collection_db_path).expanduser().resolve()
    queries_db_path = Path(queries_db_path).expanduser().resolve()
    qrels_path = Path(qrels_path).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()

    if record_batch_size <= 0:
        raise ValueError("record_batch_size must be > 0")

    dtype = np.dtype(dtype)
    if dtype not in (np.float16, np.float32, np.float64):
        raise ValueError(f"Unsupported dtype: {dtype}")

    a_path = None
    bc_path = None
    meta_path = None
    expected_meta = None
    if use_cache:
        _ensure_cache_dir(cache_dir)
        tag = (
            f"msmarco_collection_{collection_table_name}_queries_{queries_table_name}_uniqueqrels_dev"
        )
        dtype_tag = {np.float16: "f16", np.float32: "f32", np.float64: "f64"}[dtype.type]
        a_path = cache_dir / f"{tag}_a_{dtype_tag}.npy"
        bc_path = cache_dir / f"{tag}_bc_{dtype_tag}.npz"
        meta_path = cache_dir / f"{tag}_meta.json"

        expected_meta = {
            "collection_db": _fingerprint(collection_db_path),
            "collection_table": collection_table_name,
            "queries_db": _fingerprint(queries_db_path),
            "queries_table": queries_table_name,
            "qrels": _fingerprint(qrels_path),
            "dtype": str(dtype),
        }

        if not force_recompute:
            meta = _load_meta(meta_path)
            if (
                meta is not None
                and a_path.exists()
                and bc_path.exists()
                and _meta_matches(meta, expected=expected_meta)
            ):
                a = np.load(a_path, mmap_mode=mmap_mode)
                with np.load(bc_path) as payload:
                    b = payload["b"]
                    c = payload["c"]
                return a, b, c

    # ---- Recompute ----
    qids, pids = _read_qrels_unique_pairs(qrels_path)

    # Load collection: item_id -> row in a.
    with duckdb.connect(str(collection_db_path), read_only=True) as conn:
        stats = conn.execute(
            f"""
            SELECT
              COUNT(*)::BIGINT AS n,
              COALESCE(MIN(item_id), 0)::BIGINT AS min_id,
              COALESCE(MAX(item_id), -1)::BIGINT AS max_id
            FROM {collection_table_name};
            """
        ).fetchone()
        if not stats:
            raise RuntimeError("Failed to query collection table stats")
        n_rows = int(stats[0])
        min_id = int(stats[1])
        max_id = int(stats[2])
        if n_rows <= 0:
            raise ValueError(f"Empty collection table: {collection_db_path}::{collection_table_name}")
        if max_id < 0:
            raise ValueError("Collection table has no item_id")

        dim_row = conn.execute(
            f"SELECT array_length(embedding) FROM {collection_table_name} LIMIT 1;"
        ).fetchone()
        if not dim_row:
            raise RuntimeError("Failed to determine embedding dimension")
        dim = int(dim_row[0])
        if dim <= 0:
            raise ValueError(f"Invalid embedding dimension: {dim}")

        if min_id != 0:
            raise ValueError(f"Expected collection item_id to start at 0, got min_id={min_id}")

        contiguous = (max_id + 1) == n_rows
        N = n_rows if contiguous else (max_id + 1)

        if use_cache:
            assert a_path is not None
            a = np.lib.format.open_memmap(a_path, mode="w+", dtype=dtype, shape=(N, dim))
        else:
            a = np.empty((N, dim), dtype=dtype)

        reader = conn.execute(
            f"SELECT item_id, embedding FROM {collection_table_name};"
        ).fetch_record_batch(record_batch_size)

        for batch in reader:
            batch_n = batch.num_rows
            if batch_n == 0:
                continue
            item_ids = np.asarray(
                batch.column(0).to_numpy(zero_copy_only=False), dtype=np.int64
            )
            emb = _fixed_size_list_to_numpy_2d(batch.column(1), dim, dtype)

            is_consecutive = (
                item_ids[-1] - item_ids[0] + 1 == item_ids.shape[0]
                and np.all(item_ids[1:] == item_ids[:-1] + 1)
            )
            if is_consecutive:
                start = int(item_ids[0])
                a[start : start + batch_n] = emb
            else:
                a[item_ids] = emb

        _normalize_rows_inplace(a, chunk_rows=normalize_chunk_rows)
        if use_cache and hasattr(a, "flush"):
            a.flush()

    # Load dev queries embeddings for selected qids.
    k = int(qids.shape[0])
    with duckdb.connect(str(queries_db_path), read_only=False) as conn:
        dim_row = conn.execute(
            f"SELECT array_length(embedding) FROM {queries_table_name} LIMIT 1;"
        ).fetchone()
        if not dim_row:
            raise ValueError(f"Empty queries table: {queries_db_path}::{queries_table_name}")
        q_dim = int(dim_row[0])
        if q_dim != dim:
            raise ValueError(f"Dim mismatch: collection dim={dim}, query dim={q_dim}")

        conn.execute("DROP TABLE IF EXISTS selected_qids;")
        conn.execute("CREATE TEMP TABLE selected_qids(qid BIGINT PRIMARY KEY, idx BIGINT);")
        conn.executemany(
            "INSERT INTO selected_qids VALUES (?, ?);",
            [(int(qid), int(i)) for i, qid in enumerate(qids.tolist())],
        )

        arrow = conn.execute(
            f"""
            SELECT s.idx AS idx, q.embedding AS embedding
            FROM selected_qids s
            JOIN {queries_table_name} q ON q.item_id = s.qid
            ORDER BY s.idx;
            """
        ).fetch_arrow_table()

        if arrow.num_rows != k:
            missing_rows = conn.execute(
                f"""
                SELECT qid
                FROM selected_qids
                EXCEPT
                SELECT item_id AS qid
                FROM {queries_table_name}
                WHERE item_id IN (SELECT qid FROM selected_qids);
                """
            ).fetchall()
            missing = [int(r[0]) for r in missing_rows]
            raise ValueError(
                f"Missing {len(missing)} qids in queries table (example: {missing[:10]})"
            )

        emb_col = arrow.column("embedding")
        c_chunks = [
            _fixed_size_list_to_numpy_2d(emb_col.chunk(i), dim, dtype) for i in range(emb_col.num_chunks)
        ]
        c = np.concatenate(c_chunks, axis=0) if len(c_chunks) > 1 else c_chunks[0]
        if not c.flags.writeable:
            c = c.copy()
        if c.shape != (k, dim):
            raise RuntimeError(f"Unexpected c shape: {c.shape}, expected {(k, dim)}")

        _normalize_rows_inplace(c, chunk_rows=normalize_chunk_rows)

    b = pids

    if use_cache:
        assert a_path is not None and bc_path is not None and meta_path is not None and expected_meta is not None
        np.savez(bc_path, b=b, c=c)
        meta = dict(expected_meta)
        meta.update(
            {
                "a_path": str(a_path),
                "bc_path": str(bc_path),
                "a_shape": [int(a.shape[0]), int(a.shape[1])],
                "k": int(k),
            }
        )
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if mmap_mode is not None:
            a = np.load(a_path, mmap_mode=mmap_mode)

    return a, b, c


if __name__ == "__main__":
    a, b, c = load_msmarco_collection_dev_unique_qrels()
    print(f"a.shape={a.shape}")
    print(f"b.shape={b.shape}")
    print(f"c.shape={c.shape}")
