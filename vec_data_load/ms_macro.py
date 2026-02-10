from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import duckdb
import torch
from transformers import AutoModel, AutoTokenizer

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_MODEL = "sentence-transformers/msmarco-MiniLM-L6-v3"
DEFAULT_MAX_LENGTH = 512

DATASET_PRESETS = {
    "collection": {
        "tsv_path": Path("data/msmacro/collection.tsv"),
        "db_path": Path("data/msmacro/collection.duckdb"),
        "table_name": "collection_embeddings",
    },
    "queries_dev": {
        "tsv_path": Path("data/msmacro/queries.dev.tsv"),
        "db_path": Path("data/msmacro/queries.dev.duckdb"),
        "table_name": "queries_dev_embeddings",
    },
    "queries_train": {
        "tsv_path": Path("data/msmacro/queries.train.tsv"),
        "db_path": Path("data/msmacro/queries.train.duckdb"),
        "table_name": "queries_train_embeddings",
    },
    "queries_eval": {
        "tsv_path": Path("data/msmacro/queries.eval.tsv"),
        "db_path": Path("data/msmacro/queries.eval.duckdb"),
        "table_name": "queries_eval_embeddings",
    },
}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def _device_or_default(device_arg: Optional[str]) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_table(conn: duckdb.DuckDBPyConnection, table_name: str, dim: int) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            row_id BIGINT PRIMARY KEY,
            item_id BIGINT,
            embedding FLOAT[{dim}]
        );
        """
    )


def _table_row_stats(conn: duckdb.DuckDBPyConnection, table_name: str) -> Tuple[int, int]:
    """
    Returns: (row_count, max_row_id) where max_row_id is -1 if table is empty.
    """
    row = conn.execute(
        f"SELECT COUNT(*) AS n, COALESCE(MAX(row_id), -1) AS max_row_id FROM {table_name};"
    ).fetchone()
    if not row:
        return 0, -1
    return int(row[0]), int(row[1])


def _iter_tsv_rows(
    tsv_path: Path, start_row: int = 0, max_rows: Optional[int] = None
) -> Iterable[Tuple[int, int, str]]:
    with tsv_path.open("r", encoding="utf-8", errors="replace") as f:
        for row_id, line in enumerate(f):
            if row_id < start_row:
                continue
            if max_rows is not None and row_id >= max_rows:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                item_id_str, text = line.split("\t", 1)
            except ValueError:
                raise ValueError(f"Line {row_id + 1} malformed: expected two tab-separated columns")
            try:
                item_id = int(item_id_str)
            except ValueError as exc:
                raise ValueError(f"Line {row_id + 1} has non-integer id: {item_id_str}") from exc
            yield row_id, item_id, text


def _embed_texts(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    texts: Sequence[str],
    *,
    max_length: int,
    normalize: bool,
) -> List[List[float]]:
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    inference_ctx = getattr(torch, "inference_mode", torch.no_grad)
    with inference_ctx():
        output = model(**encoded)

    pooled = mean_pooling(output, encoded["attention_mask"])
    if normalize:
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.cpu().numpy().tolist()


def process_tsv_to_duckdb(
    *,
    tsv_path: Path,
    db_path: Path,
    table_name: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    batch_size: int,
    max_length: int,
    normalize: bool,
    max_rows: Optional[int],
    limit: Optional[int],
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be >= 0")

    tsv_path = tsv_path.expanduser().resolve()
    db_path = db_path.expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db_path))
    embedding_dim = int(model.config.hidden_size)
    _ensure_table(conn, table_name, embedding_dim)

    existing_rows, max_row_id = _table_row_stats(conn, table_name)
    start_row = max_row_id + 1
    if start_row != existing_rows:
        print(
            f"[{table_name}] warning: table has gaps (count={existing_rows}, max_row_id={max_row_id}); "
            "resuming from max_row_id+1"
        )

    effective_max_rows = max_rows
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        stop_row = start_row + limit
        effective_max_rows = (
            stop_row if effective_max_rows is None else min(effective_max_rows, stop_row)
        )

    if effective_max_rows is not None and start_row >= effective_max_rows:
        parts = [f"start_row={start_row}", f"stop_row={effective_max_rows}"]
        if max_rows is not None:
            parts.append(f"max_rows={max_rows}")
        if limit is not None:
            parts.append(f"limit={limit}")
        print(f"[{table_name}] nothing to do ({', '.join(parts)})")
        conn.close()
        return

    rows_iter = _iter_tsv_rows(tsv_path, start_row=start_row, max_rows=effective_max_rows)
    pbar = None
    if tqdm is not None:
        total = None
        if effective_max_rows is not None:
            total = max(0, effective_max_rows - start_row)
        pbar = tqdm(total=total, desc=table_name, unit="row")

    processed = 0
    try:
        while True:
            batch = list(itertools.islice(rows_iter, batch_size))
            if not batch:
                break

            texts = [text for _, _, text in batch]
            embeddings = _embed_texts(
                tokenizer,
                model,
                device,
                texts,
                max_length=max_length,
                normalize=normalize,
            )

            to_insert = [
                (row_id, item_id, emb) for (row_id, item_id, _), emb in zip(batch, embeddings)
            ]
            conn.executemany(
                f"""
                INSERT INTO {table_name} (row_id, item_id, embedding)
                VALUES (?, ?, ?)
                ON CONFLICT(row_id) DO NOTHING;
                """,
                to_insert,
            )
            processed += len(batch)
            if pbar is not None:
                pbar.update(len(batch))
    finally:
        if pbar is not None:
            pbar.close()

    conn.close()
    print(f"[{table_name}] processed {processed} new rows (start={start_row}, total_written={start_row + processed})")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode MS MARCO TSV (collection / queries.*) into DuckDB embeddings (resumable)."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_PRESETS.keys()),
        default="collection",
        help="预设数据集名称，决定默认的 tsv/db 路径和表名",
    )
    parser.add_argument("--tsv-path", type=Path, help="自定义 TSV 路径（默认取决于 --dataset）")
    parser.add_argument("--db-path", type=Path, help="自定义 DuckDB 路径（默认取决于 --dataset）")
    parser.add_argument("--table-name", help="DuckDB 表名（默认取决于 --dataset）")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="HuggingFace embedding 模型名")
    parser.add_argument("--device", default=None, help="torch device (默认自动选择 cuda 可用则 cuda)")
    parser.add_argument("--batch", type=int, default=128, help="每批处理条数")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="最多处理多少行（相当于 TSV 前 N 行；默认处理到文件结尾）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="本次运行最多新增处理多少行（用于分批跑；默认不限制）",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="tokenizer truncation 的 max_length",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否对 embedding 做 L2 归一化（默认开启）",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    preset = DATASET_PRESETS[args.dataset]
    tsv_path = args.tsv_path or preset["tsv_path"]
    db_path = args.db_path or preset["db_path"]
    table_name = args.table_name or preset["table_name"]

    device = _device_or_default(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.eval()
    model.to(device)

    process_tsv_to_duckdb(
        tsv_path=tsv_path,
        db_path=db_path,
        table_name=table_name,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch,
        max_length=args.max_length,
        normalize=args.normalize,
        max_rows=args.max_rows,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
