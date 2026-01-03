"""
BUPT-CBFace embedding loader (resumable).

按 `landmark.tsv` 的顺序逐行读取 (NAME + 5-point landmarks)，对齐到 112x112，
用 ArcFace 模型抽取 512-d embedding，并保存到 DuckDB（每个数据集一个库，放在各自目录下）。

为什么用 DuckDB：顺序处理过程中如果中断/报错，重新运行会自动从上次已写入的
最后一行继续。
"""

from __future__ import annotations

import argparse
import csv
import os
from itertools import islice
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import duckdb
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from insightface.model_zoo import get_model
from insightface.utils import face_align


EMBEDDING_DIM = 512
TABLE_NAME = "bupt_cbface_embeddings"


def _count_data_rows(tsv_path: Path) -> int:
    with tsv_path.open("r", encoding="utf-8", errors="replace") as f:
        line_count = sum(1 for _ in f)
    return max(0, line_count - 1)  # minus header


def _parse_name(name: str) -> Tuple[str, Optional[int]]:
    parts = name.split("/")
    if len(parts) != 2:
        return name, None
    person = parts[0]
    try:
        idx = int(parts[1])
    except ValueError:
        idx = None
    return person, idx


def _parse_pts(row: dict) -> np.ndarray:
    pts = np.array(
        [
            [float(row["PTX1"]), float(row["PTY1"])],
            [float(row["PTX2"]), float(row["PTY2"])],
            [float(row["PTX3"]), float(row["PTY3"])],
            [float(row["PTX4"]), float(row["PTY4"])],
            [float(row["PTX5"]), float(row["PTY5"])],
        ],
        dtype=np.float32,
    )
    if pts.shape != (5, 2):
        raise ValueError(f"Invalid landmark shape: {pts.shape}")
    return pts


def _compute_embedding(
    rec,
    img_path: Path,
    pts: np.ndarray,
    image_size: int,
) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    aligned = face_align.norm_crop(img, landmark=pts, image_size=image_size)
    emb = rec.get_feat(aligned).flatten()
    if emb.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Unexpected embedding dim: got {emb.shape[0]}, want {EMBEDDING_DIM}")
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb


def _ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            dataset TEXT,
            row_id BIGINT,
            name TEXT,
            person TEXT,
            img_idx INTEGER,
            vec FLOAT[{EMBEDDING_DIM}],
            PRIMARY KEY (dataset, row_id)
        );
        """
    )


def _last_row_id(conn: duckdb.DuckDBPyConnection, dataset: str) -> int:
    row = conn.execute(
        f"SELECT COALESCE(MAX(row_id), 0) FROM {TABLE_NAME} WHERE dataset=?;",
        [dataset],
    ).fetchone()
    return int(row[0]) if row else 0


def _iter_tsv_rows(tsv_path: Path, start_row_id: int, limit: Optional[int]) -> Iterable[Tuple[int, dict]]:
    with tsv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        if start_row_id <= 1:
            it = reader
            row_id_start = 1
        else:
            it = islice(reader, start_row_id - 1, None)
            row_id_start = start_row_id

        if limit is not None:
            it = islice(it, 0, limit)

        for offset, row in enumerate(it, start=row_id_start):
            yield offset, row


def process_dataset(
    conn: duckdb.DuckDBPyConnection,
    dataset_dir: Path,
    rec,
    image_size: int,
    limit: Optional[int],
) -> None:
    dataset_dir = dataset_dir.resolve()
    dataset = dataset_dir.name
    tsv_path = dataset_dir / "landmark.tsv"
    images_dir = dataset_dir / "images"

    if not tsv_path.exists():
        raise FileNotFoundError(f"[{dataset}] missing landmark.tsv: {tsv_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"[{dataset}] missing images/: {images_dir}")

    total_rows = _count_data_rows(tsv_path)
    last_row = _last_row_id(conn, dataset)
    start_row = last_row + 1

    if last_row > total_rows:
        raise ValueError(
            f"[{dataset}] db last_row={last_row} > tsv rows={total_rows}, "
            "dataset/DB可能不匹配 (建议换 db_path 或清空表)"
        )

    remaining = max(0, total_rows - last_row)
    if limit is not None:
        remaining = min(remaining, limit)

    rows_iter = _iter_tsv_rows(tsv_path, start_row_id=start_row, limit=limit)
    if tqdm is not None:
        rows_iter = tqdm(rows_iter, total=remaining, desc=f"{dataset}", unit="img")

    for row_id, row in rows_iter:
        name = row["NAME"]
        img_path = images_dir / f"{name}.jpg"
        person, img_idx = _parse_name(name)
        pts = _parse_pts(row)
        emb = _compute_embedding(rec, img_path=img_path, pts=pts, image_size=image_size)

        conn.execute(
            f"""
            INSERT INTO {TABLE_NAME} (dataset, row_id, name, person, img_idx, vec)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING;
            """,
            [dataset, row_id, name, person, img_idx, emb.tolist()],
        )


def _default_data_dirs() -> Sequence[str]:
    candidates = ["data/BUPT-CBFace-12", "data/BUPT-CBFace-50"]
    return [p for p in candidates if Path(p).exists()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="BUPT-CBFace landmark.tsv -> ArcFace embeddings -> DuckDB")
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=_default_data_dirs(),
        help="数据目录列表 (每个目录需包含 landmark.tsv + images/)",
    )
    parser.add_argument(
        "--db-name",
        default="arcface_embeddings.duckdb",
        help="每个数据目录下生成的 DuckDB 文件名 (用于断点续跑)",
    )
    parser.add_argument(
        "--model-path",
        default=os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx"),
        help="ArcFace ONNX 模型路径 (默认 buffalo_l/w600k_r50.onnx)",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["CPUExecutionProvider"],
        help="onnxruntime providers (例如 CPUExecutionProvider 或 CUDAExecutionProvider)",
    )
    parser.add_argument("--ctx-id", type=int, default=0, help="insightface ctx_id (CPU 时通常无所谓)")
    parser.add_argument("--image-size", type=int, default=112, help="对齐输出尺寸 (ArcFace 通常 112)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="每个数据集最多处理多少行 (调试用，默认处理到文件结尾)",
    )

    args = parser.parse_args(argv)

    if not args.data_dirs:
        raise SystemExit("No data dirs found. 请传 --data-dirs ... 或确保 data/ 下有 BUPT-CBFace-12/50")

    rec = get_model(args.model_path, providers=args.providers)
    rec.prepare(ctx_id=args.ctx_id)

    for data_dir in args.data_dirs:
        dataset_dir = Path(data_dir).resolve()
        db_path = dataset_dir / args.db_name

        conn = duckdb.connect(str(db_path))
        _ensure_schema(conn)
        process_dataset(
            conn=conn,
            dataset_dir=dataset_dir,
            rec=rec,
            image_size=args.image_size,
            limit=args.limit,
        )
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
