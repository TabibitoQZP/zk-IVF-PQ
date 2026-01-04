from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import duckdb
import numpy as np


EMBEDDING_DIM = 512
TABLE_NAME = "bupt_cbface_embeddings"

PathLike = Union[str, Path]


def _resolve_db_path(db_dir_or_path: PathLike, db_name: str) -> Path:
    p = Path(db_dir_or_path).expanduser()
    db_path = (p / db_name) if p.is_dir() else p
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")
    return db_path.resolve()


def _fixed_size_list_to_numpy_2d(arr, dim: int, dtype: np.dtype) -> np.ndarray:
    # arr: pyarrow.FixedSizeListArray
    flat = arr.values.to_numpy(zero_copy_only=False)
    out = flat.reshape((len(arr), dim))
    if out.dtype != dtype:
        out = out.astype(dtype, copy=False)
    return out


def sample_bupt_cbface_queries_db_ground_truth(
    db_dir: PathLike,
    *,
    db_name: str = "arcface_embeddings.duckdb",
    num_queries: int = 1024,
    ground_truth_k: int = 11,
    seed: int = 0,
    batch_size: int = 16384,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从单个 BUPT-CBFace DuckDB 中构造 ANN 评测用的 (queries, db, ground_truth)。

    假设 DuckDB 里有 n 条 embedding：
      1) 随机取 num_queries 个不同的 person，每个 person 随机取 1 个 embedding -> queries (Q,512)
      2) 删除这些 query 行后，剩余 embedding 随机打乱 -> db (n-Q,512)
      3) 对每条 query，统计其同 person 的 embedding 在 db 里的索引位置，
         取前 ground_truth_k 个 -> ground_truth (Q,K)

    说明：
      - BUPT-CBFace-12 每个 person 12 张：默认 ground_truth_k=11 刚好是全部剩余同人样本
      - BUPT-CBFace-50 每个 person 50 张：默认只取 11 个；如需全部可传 ground_truth_k=49
    """
    if num_queries <= 0:
        raise ValueError("num_queries must be > 0")
    if ground_truth_k <= 0:
        raise ValueError("ground_truth_k must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    db_path = _resolve_db_path(db_dir, db_name=db_name)

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        n_total = int(conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};").fetchone()[0])
        if n_total < num_queries:
            raise ValueError(
                f"Not enough rows: n_total={n_total} < num_queries={num_queries}"
            )

        eligible = conn.execute(
            f"""
            SELECT person
            FROM {TABLE_NAME}
            GROUP BY person
            HAVING COUNT(*) >= ?
            ORDER BY person;
            """,
            [ground_truth_k + 1],
        ).fetchall()
        eligible_persons = [r[0] for r in eligible]
        if len(eligible_persons) < num_queries:
            raise ValueError(
                f"Not enough eligible persons: {len(eligible_persons)} < num_queries={num_queries} "
                f"(need COUNT(person) >= {ground_truth_k + 1})"
            )

        rng = np.random.default_rng(seed)
        selected_persons = rng.choice(
            eligible_persons, size=num_queries, replace=False
        ).tolist()

        conn.execute("DROP TABLE IF EXISTS selected_persons;")
        conn.execute("CREATE TEMP TABLE selected_persons(person TEXT, q_idx INTEGER);")
        conn.executemany(
            "INSERT INTO selected_persons VALUES (?, ?);",
            [(p, i) for i, p in enumerate(selected_persons)],
        )

        # Deterministic per-person random pick in DuckDB.
        duck_seed = float(np.random.default_rng(seed + 1).random() * 2.0 - 1.0)
        conn.execute("SELECT setseed(?);", [duck_seed])

        arrow = conn.execute(
            f"""
            SELECT sp.q_idx AS q_idx, e.row_id AS row_id, e.vec AS vec
            FROM (
              SELECT person, row_id, vec,
                     row_number() OVER (PARTITION BY person ORDER BY random()) AS rn
              FROM {TABLE_NAME}
              WHERE person IN (SELECT person FROM selected_persons)
            ) e
            JOIN selected_persons sp ON e.person = sp.person
            WHERE e.rn = 1
            ORDER BY sp.q_idx;
            """
        ).fetch_arrow_table()

        if arrow.num_rows != num_queries:
            raise ValueError(f"Expected {num_queries} query rows, got {arrow.num_rows}")

        row_id = np.asarray(
            arrow.column("row_id").to_numpy(zero_copy_only=False), dtype=np.int64
        )
        vec_col = arrow.column("vec")
        vec_chunks = [
            _fixed_size_list_to_numpy_2d(vec_col.chunk(i), EMBEDDING_DIM, dtype)
            for i in range(vec_col.num_chunks)
        ]
        queries = (
            np.concatenate(vec_chunks, axis=0) if len(vec_chunks) > 1 else vec_chunks[0]
        )

        conn.execute("DROP TABLE IF EXISTS query_rows;")
        conn.execute("CREATE TEMP TABLE query_rows(row_id BIGINT PRIMARY KEY);")
        conn.executemany(
            "INSERT INTO query_rows VALUES (?);",
            [(int(rid),) for rid in row_id.tolist()],
        )

        n_db = n_total - num_queries
        db = np.empty((n_db, EMBEDDING_DIM), dtype=dtype)
        db_q_idx = np.empty((n_db,), dtype=np.int32)

        perm = rng.permutation(n_db).astype(np.int64, copy=False)

        reader = conn.execute(
            f"""
            SELECT e.vec AS vec, COALESCE(sp.q_idx, -1) AS q_idx
            FROM {TABLE_NAME} e
            LEFT JOIN selected_persons sp ON e.person = sp.person
            LEFT JOIN query_rows qr ON e.row_id = qr.row_id
            WHERE qr.row_id IS NULL;
            """
        ).fetch_record_batch(batch_size)

        write_pos = 0
        for batch in reader:
            batch_n = batch.num_rows
            if batch_n == 0:
                continue

            vecs = _fixed_size_list_to_numpy_2d(batch.column(0), EMBEDDING_DIM, dtype)
            q_idx = (
                batch.column(1)
                .to_numpy(zero_copy_only=False)
                .astype(np.int32, copy=False)
            )

            target = perm[write_pos : write_pos + batch_n]
            if target.shape[0] != batch_n:
                raise ValueError("Internal error: permutation slice size mismatch")

            db[target] = vecs
            db_q_idx[target] = q_idx
            write_pos += batch_n

        if write_pos != n_db:
            raise ValueError(
                f"Internal error: expected {n_db} db rows, got {write_pos}"
            )

        relevant_pos = np.nonzero(db_q_idx >= 0)[0]
        relevant_ids = db_q_idx[relevant_pos]
        order = np.argsort(relevant_ids, kind="stable")
        relevant_pos = relevant_pos[order]
        relevant_ids = relevant_ids[order]

        unique_ids, starts, counts = np.unique(
            relevant_ids, return_index=True, return_counts=True
        )
        if (
            unique_ids.shape[0] != num_queries
            or unique_ids.min() != 0
            or unique_ids.max() != num_queries - 1
        ):
            raise ValueError("Internal error: query-person ids missing in db")

        ground_truth = np.empty((num_queries, ground_truth_k), dtype=np.int64)
        for qid, start, cnt in zip(
            unique_ids.tolist(), starts.tolist(), counts.tolist()
        ):
            if cnt < ground_truth_k:
                raise ValueError(
                    f"Person q_idx={qid} has only {cnt} matches in db, need >= {ground_truth_k}"
                )
            ground_truth[qid] = relevant_pos[start : start + ground_truth_k]

        return queries, db, ground_truth
    finally:
        conn.close()


if __name__ == "__main__":
    queries, db, gt = sample_bupt_cbface_queries_db_ground_truth("data/BUPT-CBFace-12")
    print(queries.shape, db.shape, gt.shape)
    queries, db, gt = sample_bupt_cbface_queries_db_ground_truth(
        "data/BUPT-CBFace-50", ground_truth_k=49
    )
    print(queries.shape, db.shape, gt.shape)
