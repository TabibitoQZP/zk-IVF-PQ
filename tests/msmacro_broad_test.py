from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

# Allow running as `python tests/msmacro_broad_test.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import faiss  # noqa: E402

from vec_data_load.ms_macro_load import (  # noqa: E402
    _read_qrels_unique_pairs,
    load_msmarco_collection_dev_unique_qrels,
)


SUPPORTED_KINDS = {"ivf_flat", "ivf_pq", "hnsw", "hnsw_pq"}
SUPPORTED_METRICS = {"ip": faiss.METRIC_INNER_PRODUCT, "l2": faiss.METRIC_L2}


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _fingerprint(path: Path) -> Dict[str, object]:
    st = path.stat()
    return {"path": str(path.resolve()), "mtime_ns": int(st.st_mtime_ns), "size": int(st.st_size)}


def _stable_json_dumps(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _short_hash(obj: object, *, n: int = 10) -> str:
    h = hashlib.sha1(_stable_json_dumps(obj).encode("utf-8")).hexdigest()
    return h[:n]


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


def _maybe_rm_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


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


def _ensure_float32_contiguous(x: np.ndarray, *, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={x.shape}")
    if not x.flags.c_contiguous:
        x = np.ascontiguousarray(x)
    return x


def _validate_config(cfg: dict) -> dict:
    if "top_k" not in cfg:
        raise ValueError("Config missing required key: top_k")
    top_k = int(cfg["top_k"])
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    metric_name = str(cfg.get("metric", "l2")).lower()
    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric: {metric_name} (expected one of {sorted(SUPPORTED_METRICS)})")

    save_sims = bool(cfg.get("save_sims", False))

    series = cfg.get("series")
    if not isinstance(series, list) or not series:
        raise ValueError("Config must contain non-empty list key: series")

    validated_series = []
    for i, s in enumerate(series):
        if not isinstance(s, dict):
            raise ValueError(f"series[{i}] must be an object")
        name = str(s.get("name", "")).strip()
        if not name:
            raise ValueError(f"series[{i}] missing non-empty 'name'")
        kind = str(s.get("kind", "")).strip().lower()
        if kind not in SUPPORTED_KINDS:
            raise ValueError(
                f"series[{i}] kind={kind!r} unsupported (expected one of {sorted(SUPPORTED_KINDS)})"
            )
        build = s.get("build")
        if not isinstance(build, dict):
            raise ValueError(f"series[{i}] missing 'build' object")
        sweep = s.get("sweep")
        if not isinstance(sweep, dict) or len(sweep) != 1:
            raise ValueError(f"series[{i}] 'sweep' must be an object with exactly 1 key")

        (sweep_key, sweep_values_raw), *_ = sweep.items()
        sweep_key = str(sweep_key)
        if not isinstance(sweep_values_raw, list) or not sweep_values_raw:
            raise ValueError(f"series[{i}] sweep.{sweep_key} must be a non-empty list")
        sweep_values = [int(v) for v in sweep_values_raw]
        if any(v <= 0 for v in sweep_values):
            raise ValueError(f"series[{i}] sweep.{sweep_key} contains non-positive value(s)")

        if kind in ("ivf_flat", "ivf_pq") and sweep_key != "nprobe":
            raise ValueError(f"series[{i}] kind={kind} must sweep 'nprobe'")
        if kind in ("hnsw", "hnsw_pq") and sweep_key != "efSearch":
            raise ValueError(f"series[{i}] kind={kind} must sweep 'efSearch'")

        validated_series.append(
            {
                "name": name,
                "kind": kind,
                "build": build,
                "sweep_key": sweep_key,
                "sweep_values": sweep_values,
            }
        )

    return {
        "top_k": top_k,
        "metric_name": metric_name,
        "metric": SUPPORTED_METRICS[metric_name],
        "save_sims": save_sims,
        "series": validated_series,
    }


def _example_config() -> dict:
    return {
        "top_k": 1000,
        "metric": "l2",
        "save_sims": False,
        "series": [
            {
                "name": "ivf_flat_nlist4096",
                "kind": "ivf_flat",
                "build": {"nlist": 4096, "seed": 1234},
                "sweep": {"nprobe": [1, 2, 4, 8, 16, 32, 64]},
            },
            {
                "name": "ivf_pq_nlist4096_m8",
                "kind": "ivf_pq",
                "build": {"nlist": 4096, "m": 8, "nbits": 8, "seed": 1234},
                "sweep": {"nprobe": [1, 2, 4, 8, 16, 32, 64]},
            },
            {
                "name": "hnsw_flat_m32",
                "kind": "hnsw",
                "build": {"M": 32, "efConstruction": 200},
                "sweep": {"efSearch": [16, 32, 64, 128, 256, 512]},
            },
            {
                "name": "hnsw_pq_m32_pqm16",
                "kind": "hnsw_pq",
                "build": {"M": 32, "m": 16, "nbits": 8, "efConstruction": 200},
                "sweep": {"efSearch": [16, 32, 64, 128, 256, 512]},
            },
        ],
    }


def _check_unit_norm_samples(
    x: np.ndarray,
    *,
    name: str,
    tol: float = 1e-3,
    block: int = 1024,
) -> None:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={x.shape}")
    n = int(x.shape[0])
    if n == 0:
        raise ValueError(f"{name} is empty")
    block = max(1, int(block))

    slices: List[Tuple[int, int]] = [(0, min(n, block))]
    if n > block:
        mid = max(0, (n // 2) - (block // 2))
        slices.append((mid, min(n, mid + block)))
        tail = max(0, n - block)
        slices.append((tail, n))

    worst = 0.0
    for start, end in slices:
        block_x = np.asarray(x[start:end], dtype=np.float32)
        norms = np.sqrt(np.einsum("ij,ij->i", block_x, block_x))
        dev = np.abs(norms - 1.0)
        worst = max(worst, float(dev.max(initial=0.0)))

    if worst > tol:
        raise ValueError(
            f"{name} does not appear L2-normalized: worst |norm-1|={worst:.6g} > tol={tol}. "
            "Regenerate the cache with row-normalization (or inspect the DuckDB embeddings)."
        )


def _get_metric(row: dict, *, k: int, field: str) -> float:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Row missing metrics dict")
    key = str(int(k))
    if key in metrics:
        m = metrics[key]
    elif k in metrics:
        m = metrics[k]
    else:
        raise KeyError(f"metrics missing k={k}")
    if not isinstance(m, dict) or field not in m:
        raise KeyError(f"metrics[{k}] missing field {field!r}")
    return float(m[field])


def _lazy_import_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Plotting requires matplotlib. Install it (e.g. `pip install matplotlib`) "
            "or run without --plot/--plot-only."
        ) from e
    return plt


def plot_from_report(
    *,
    report: dict,
    out_dir: Path,
    plot_dir: Optional[Path] = None,
    xscale: str = "log",
    fmt: str = "png",
) -> List[Path]:
    """
    Plot speed-accuracy curves using report.json:
      - MRR@10 vs QPS
      - hit@1000 vs QPS

    All series are drawn on the same axes for each figure.
    """
    if xscale not in ("linear", "log"):
        raise ValueError("--plot-xscale must be 'linear' or 'log'")
    fmt = fmt.strip().lower().lstrip(".") or "png"

    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("report has no rows to plot")

    # group rows by series name (4 schemes in our default config)
    by_name: Dict[str, List[dict]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name", "")).strip() or "unknown"
        by_name.setdefault(name, []).append(r)

    # Sort each curve by x (QPS), then by sweep_value for stability.
    for name in by_name:
        by_name[name].sort(
            key=lambda r: (
                float(r.get("qps_total") or 0.0),
                int(r.get("sweep_value") or 0),
            )
        )

    out_dir = out_dir.expanduser().resolve()
    plot_dir = (out_dir / "plots") if plot_dir is None else plot_dir.expanduser().resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt = _lazy_import_pyplot()

    def make_fig(*, k: int, field: str, ylabel: str, out_name: str) -> Optional[Path]:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for name, lst in by_name.items():
            xs: List[float] = []
            ys: List[float] = []
            for r in lst:
                qps = r.get("qps_total")
                if not isinstance(qps, (int, float)) or qps <= 0:
                    continue
                try:
                    y = _get_metric(r, k=k, field=field)
                except Exception:
                    continue
                xs.append(float(qps))
                ys.append(float(y))
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=4, label=name)

        if not ax.lines:
            plt.close(fig)
            return None

        ax.set_xlabel("QPS (queries/sec)")
        ax.set_ylabel(ylabel)
        if xscale == "log":
            ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        out_path = plot_dir / f"{out_name}.{fmt}"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    paths: List[Path] = []

    for k in (10, 100, 1000):
        out = make_fig(k=k, field="hit", ylabel=f"Hit@{k}", out_name=f"hit{k}_vs_qps")
        if out is not None:
            paths.append(out)

    for k in (10, 100, 1000):
        out = make_fig(k=k, field="mrr", ylabel=f"MRR@{k}", out_name=f"mrr{k}_vs_qps")
        if out is not None:
            paths.append(out)

    if not paths:
        raise ValueError(
            "No plots generated. Ensure report.json contains qps_total and metrics for k in {10,100,1000}."
        )
    return paths


def _make_index(
    *,
    kind: str,
    d: int,
    metric: int,
    build: dict,
) -> Tuple[faiss.Index, bool, int]:
    """
    Returns: (index, needs_train, train_size_hint)
    """
    if kind == "ivf_flat":
        nlist = int(build["nlist"])
        quantizer = faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, metric)
        index.cp.seed = int(build.get("seed", index.cp.seed))
        index.cp.max_points_per_centroid = int(build.get("max_points_per_centroid", index.cp.max_points_per_centroid))
        train_size_hint = int(build.get("train_size", max(nlist, nlist * index.cp.max_points_per_centroid)))
        return index, True, train_size_hint

    if kind == "ivf_pq":
        nlist = int(build["nlist"])
        m = int(build["m"])
        nbits = int(build.get("nbits", 8))
        if m <= 0:
            raise ValueError("ivf_pq build.m must be > 0")
        if d % m != 0:
            raise ValueError(f"ivf_pq requires d % m == 0 (got d={d}, m={m})")
        if not (1 <= nbits <= 24):
            raise ValueError("ivf_pq build.nbits must be in [1, 24]")
        quantizer = faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, metric)
        index.cp.seed = int(build.get("seed", index.cp.seed))
        index.cp.max_points_per_centroid = int(build.get("max_points_per_centroid", index.cp.max_points_per_centroid))
        train_size_hint = int(build.get("train_size", max(nlist, nlist * index.cp.max_points_per_centroid)))
        return index, True, train_size_hint

    if kind == "hnsw":
        M = int(build["M"])
        index = faiss.IndexHNSWFlat(d, M, metric)
        index.hnsw.efConstruction = int(build.get("efConstruction", index.hnsw.efConstruction))
        return index, False, 0

    if kind == "hnsw_pq":
        M = int(build["M"])
        pq_m = int(build["m"])
        nbits = int(build.get("nbits", 8))
        if pq_m <= 0:
            raise ValueError("hnsw_pq build.m must be > 0")
        if d % pq_m != 0:
            raise ValueError(f"hnsw_pq requires d % m == 0 (got d={d}, m={pq_m})")
        if not (1 <= nbits <= 24):
            raise ValueError("hnsw_pq build.nbits must be in [1, 24]")
        index = faiss.IndexHNSWPQ(d, pq_m, M, nbits, metric)
        index.hnsw.efConstruction = int(build.get("efConstruction", index.hnsw.efConstruction))
        train_size_hint = int(build.get("train_size", 200_000))
        return index, True, train_size_hint

    raise ValueError(f"Unsupported kind: {kind}")


def _build_or_load_index(
    *,
    out_dir: Path,
    series_id: str,
    expected_meta: dict,
    kind: str,
    d: int,
    metric: int,
    build: dict,
    base: np.ndarray,
    batch_add: int,
    threads: Optional[int],
    force_rebuild: bool,
) -> faiss.Index:
    out_dir = out_dir.expanduser().resolve()
    index_dir = out_dir / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / f"{series_id}.faiss"
    meta_path = index_dir / f"{series_id}.json"

    if not force_rebuild and index_path.exists() and meta_path.exists():
        try:
            meta = _load_json(meta_path)
        except Exception:
            meta = None
        if isinstance(meta, dict) and meta.get("inputs") == expected_meta:
            idx = faiss.read_index(str(index_path))
            return idx

    if index_path.exists():
        index_path.unlink()
    if meta_path.exists():
        meta_path.unlink()

    if threads is not None:
        faiss.omp_set_num_threads(int(threads))

    index, needs_train, train_size_hint = _make_index(kind=kind, d=d, metric=metric, build=build)

    n = int(base.shape[0])
    if kind in ("ivf_flat", "ivf_pq"):
        nlist = int(build["nlist"])
        if nlist <= 0:
            raise ValueError("IVF build.nlist must be > 0")
        if n < nlist:
            raise ValueError(f"Database too small for IVF: n={n} < nlist={nlist}")
    base_f32 = _ensure_float32_contiguous(base, name="base")

    t0 = time.time()
    train_s = 0.0
    add_s = 0.0

    if needs_train and not index.is_trained:
        train_size = min(n, max(int(train_size_hint), 1))
        train_x = base_f32[:train_size]
        t_train0 = time.time()
        index.train(train_x)
        train_s = time.time() - t_train0

    t_add0 = time.time()
    if batch_add <= 0:
        raise ValueError("batch_add must be > 0")
    for start in tqdm(range(0, n, batch_add), desc=f"add[{series_id}]", unit="vec"):
        block = base_f32[start : start + batch_add]
        if block.size == 0:
            continue
        index.add(block)
    add_s = time.time() - t_add0

    build_s = time.time() - t0

    tmp_index_path = index_path.with_suffix(index_path.suffix + ".tmp")
    if tmp_index_path.exists():
        tmp_index_path.unlink()
    faiss.write_index(index, str(tmp_index_path))
    tmp_index_path.replace(index_path)

    meta = {
        "created_at": _now_tag(),
        "inputs": expected_meta,
        "timing_s": {"train": train_s, "add": add_s, "total": build_s},
        "faiss_version": getattr(faiss, "__version__", "unknown"),
    }
    _atomic_write_json(meta_path, meta)
    return index


def _open_or_create_memmap(
    *,
    path: Path,
    dtype: np.dtype,
    shape: Tuple[int, ...],
    fill_value,
    force_recompute: bool,
) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    if force_recompute and path.exists():
        path.unlink()
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


def _compute_rank_1based(topk_ids: np.ndarray, gt_pid: np.ndarray) -> np.ndarray:
    """
    Returns: (Q,) int16 with 0=miss, 1..K=rank (1-based).
    """
    match = topk_ids == gt_pid[:, None]
    any_match = match.any(axis=1)
    out = np.zeros((topk_ids.shape[0],), dtype=np.int16)
    if any_match.any():
        out[any_match] = match.argmax(axis=1)[any_match].astype(np.int16) + 1
    return out


def _run_or_load_search(
    *,
    out_dir: Path,
    series_id: str,
    kind: str,
    sweep_key: str,
    sweep_value: int,
    top_k: int,
    save_sims: bool,
    index: faiss.Index,
    queries: np.ndarray,
    qids: np.ndarray,
    gt_pids: np.ndarray,
    expected_meta: dict,
    batch_queries: int,
    threads: Optional[int],
    force_recompute: bool,
) -> Path:
    out_dir = out_dir.expanduser().resolve()
    run_dir = out_dir / "runs" / series_id / f"{sweep_key}{int(sweep_value)}_top{int(top_k)}"

    if force_recompute:
        _maybe_rm_tree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_path = run_dir / "meta.json"
    qids_path = run_dir / "qids.npy"
    gt_pid_path = run_dir / "gt_pid.npy"
    topk_ids_path = run_dir / "topk_ids.npy"
    topk_sims_path = run_dir / "topk_sims.npy"
    gt_rank_path = run_dir / "gt_rank.npy"

    expected_meta = dict(expected_meta)
    shapes = expected_meta.get("shapes") if isinstance(expected_meta.get("shapes"), dict) else {}
    shapes = dict(shapes)
    shapes.update(
        {
            "queries": [int(queries.shape[0]), int(queries.shape[1])],
            "qids": [int(qids.shape[0])],
            "gt_pid": [int(gt_pids.shape[0])],
        }
    )
    expected_meta.update(
        {
            "series_id": series_id,
            "kind": kind,
            "sweep_key": sweep_key,
            "sweep_value": int(sweep_value),
            "top_k": int(top_k),
            "save_sims": bool(save_sims),
            "shapes": shapes,
        }
    )

    files_ok = (
        meta_path.exists()
        and topk_ids_path.exists()
        and gt_rank_path.exists()
        and qids_path.exists()
        and gt_pid_path.exists()
        and (topk_sims_path.exists() if save_sims else True)
    )

    meta_ok = False
    if meta_path.exists():
        try:
            meta = _load_json(meta_path)
        except Exception:
            meta = None
        meta_ok = isinstance(meta, dict) and meta.get("inputs") == expected_meta
    else:
        meta = None

    if files_ok and meta_ok:
        gt_rank = np.load(gt_rank_path, mmap_mode="r")
        remaining = int((gt_rank == -1).sum())
        if remaining == 0:
            return run_dir

    # If config changed, discard old cache and start fresh.
    if not meta_ok:
        for p in (meta_path, qids_path, gt_pid_path, topk_ids_path, topk_sims_path, gt_rank_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    # Write qids / gt once (small).
    np.save(qids_path, qids.astype(np.int64, copy=False))
    np.save(gt_pid_path, gt_pids.astype(np.int32, copy=False))

    q = int(queries.shape[0])
    if batch_queries <= 0:
        raise ValueError("batch_queries must be > 0")

    topk_ids = _open_or_create_memmap(
        path=topk_ids_path,
        dtype=np.int32,
        shape=(q, int(top_k)),
        fill_value=-1,
        force_recompute=False,
    )
    topk_sims = None
    if save_sims:
        topk_sims = _open_or_create_memmap(
            path=topk_sims_path,
            dtype=np.float32,
            shape=(q, int(top_k)),
            fill_value=np.nan,
            force_recompute=False,
        )
    gt_rank = _open_or_create_memmap(
        path=gt_rank_path,
        dtype=np.int16,
        shape=(q,),
        fill_value=-1,
        force_recompute=False,
    )

    if threads is not None:
        faiss.omp_set_num_threads(int(threads))

    queries_f32 = _ensure_float32_contiguous(queries, name="queries")

    # Set search param for this run.
    if kind in ("ivf_flat", "ivf_pq"):
        index.nprobe = int(sweep_value)
    elif kind in ("hnsw", "hnsw_pq"):
        index.hnsw.efSearch = int(sweep_value)
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    searched_rows = 0
    search_s = 0.0

    for start in tqdm(range(0, q, batch_queries), desc=f"search[{series_id}:{sweep_key}={sweep_value}]", unit="q"):
        end = min(q, start + batch_queries)
        rank_block = np.asarray(gt_rank[start:end])
        missing = rank_block == -1
        if not missing.any():
            continue

        rows = np.arange(start, end, dtype=np.int64)[missing]
        q_block = queries_f32[rows]
        gt_block = gt_pids[rows].astype(np.int32, copy=False)

        t0 = time.time()
        sims, ids = index.search(q_block, int(top_k))
        dt = time.time() - t0
        search_s += dt
        searched_rows += int(q_block.shape[0])

        ids32 = ids.astype(np.int32, copy=False)
        topk_ids[rows] = ids32
        if topk_sims is not None:
            topk_sims[rows] = sims.astype(np.float32, copy=False)

        rank = _compute_rank_1based(ids32, gt_block)
        gt_rank[rows] = rank

        if hasattr(topk_ids, "flush"):
            topk_ids.flush()
        if topk_sims is not None and hasattr(topk_sims, "flush"):
            topk_sims.flush()
        if hasattr(gt_rank, "flush"):
            gt_rank.flush()

    # Load previous cumulative timings if present.
    prev_meta = None
    if meta_path.exists():
        try:
            prev_meta = _load_json(meta_path)
        except Exception:
            prev_meta = None
    prev_search_s_total = float(prev_meta.get("timing_s", {}).get("search_total", 0.0)) if isinstance(prev_meta, dict) else 0.0
    prev_rows_total = int(prev_meta.get("timing_s", {}).get("searched_rows_total", 0)) if isinstance(prev_meta, dict) else 0

    search_s_total = prev_search_s_total + float(search_s)
    rows_total = prev_rows_total + int(searched_rows)
    qps_total = float(rows_total / search_s_total) if search_s_total > 0 and rows_total > 0 else None

    remaining = int((np.asarray(gt_rank) == -1).sum())
    meta = {
        "created_at": _now_tag(),
        "inputs": expected_meta,
        "timing_s": {
            "search_last": float(search_s),
            "searched_rows_last": int(searched_rows),
            "search_total": float(search_s_total),
            "searched_rows_total": int(rows_total),
            "qps_total": qps_total,
            "remaining": remaining,
        },
        "faiss_version": getattr(faiss, "__version__", "unknown"),
    }
    _atomic_write_json(meta_path, meta)
    return run_dir


def _metrics_from_rank(rank_1based: np.ndarray, *, ks: Sequence[int]) -> Dict[int, Dict[str, float]]:
    rank = np.asarray(rank_1based)
    if rank.ndim != 1:
        raise ValueError(f"rank must be 1D, got shape={rank.shape}")
    if (rank == -1).any():
        raise ValueError("rank contains -1 (uncomputed). Run search first or remove cache.")

    out: Dict[int, Dict[str, float]] = {}
    for k in ks:
        k = int(k)
        in_k = (rank != 0) & (rank <= k)
        hit = float(in_k.mean())

        rr = np.zeros(rank.shape[0], dtype=np.float64)
        rr[in_k] = 1.0 / rank[in_k].astype(np.float64)
        mrr = float(rr.mean())

        nd = np.zeros(rank.shape[0], dtype=np.float64)
        nd[in_k] = 1.0 / np.log2(rank[in_k].astype(np.float64) + 1.0)
        ndcg = float(nd.mean())

        out[k] = {"hit": hit, "mrr": mrr, "ndcg": ndcg}
    return out


def _load_msmarco_unique_dev(
    *,
    collection_db_path: Path,
    queries_db_path: Path,
    qrels_path: Path,
    cache_dir: Path,
    max_db: Optional[int],
    max_queries: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
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

    db_total = int(a.shape[0])
    queries_total = int(qids.shape[0])

    base = a
    queries = c
    gt_pids = b

    dropped_max_db = 0
    if max_db is not None:
        max_db = int(max_db)
        if max_db <= 0:
            raise ValueError("--max-db must be > 0")
        keep = gt_pids < max_db
        dropped_max_db = int((~keep).sum())
        base = base[:max_db]
        qids = qids[keep]
        queries = queries[keep]
        gt_pids = gt_pids[keep]

    dropped_max_queries = 0
    if max_queries is not None:
        max_q = int(max_queries)
        if max_q <= 0:
            raise ValueError("--max-queries must be > 0")
        dropped_max_queries = max(0, int(qids.shape[0]) - max_q)
        qids = qids[:max_q]
        queries = queries[:max_q]
        gt_pids = gt_pids[:max_q]

    base = np.asarray(base, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    slice_info = {
        "db_total": db_total,
        "db_used": int(base.shape[0]),
        "queries_total": queries_total,
        "queries_used": int(qids.shape[0]),
        "dropped_max_db": int(dropped_max_db),
        "dropped_max_queries": int(dropped_max_queries),
    }
    return base, queries, qids, gt_pids, slice_info


def _iter_selected_series(series: Sequence[dict], only: Optional[Sequence[str]]) -> Iterable[dict]:
    if not only:
        yield from series
        return
    only_set = {s.strip() for s in only if s.strip()}
    for s in series:
        if s["name"] in only_set:
            yield s


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="MS MARCO broad ANN baseline runner (FAISS).")
    p.add_argument("--config", type=str, default="", help="Path to JSON config (A方案).")
    p.add_argument("--out-dir", type=str, default="data/msmacro/broad_test", help="Output directory.")
    p.add_argument("--collection-db", type=str, default="data/msmacro/collection.duckdb")
    p.add_argument("--queries-db", type=str, default="data/msmacro/queries.dev.duckdb")
    p.add_argument("--qrels", type=str, default="data/msmacro/qrels.dev.tsv")
    p.add_argument("--cache-dir", type=str, default="data/msmacro/cache")

    p.add_argument("--max-db", type=int, default=None)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument(
        "--allow-filtered-queries",
        action="store_true",
        help="Allow dropping queries when --max-db excludes their gt_pid (otherwise abort).",
    )

    p.add_argument("--batch-add", type=int, default=65536, help="Vectors per add() chunk when building index.")
    p.add_argument("--batch-queries", type=int, default=64, help="Queries per search() batch.")
    p.add_argument("--threads", type=int, default=None, help="faiss omp threads (optional).")

    p.add_argument("--force-rebuild-index", action="store_true")
    p.add_argument("--force-search", action="store_true")
    p.add_argument("--only-series", action="append", default=[], help="Run only matching series.name (repeatable).")
    p.add_argument("--dry-run", action="store_true", help="Print planned runs and exit.")

    p.add_argument(
        "--report-ks",
        type=str,
        default="1,5,10,20,50,100,200,500,1000",
        help="Comma-separated ks for reporting (must be <= top_k).",
    )
    p.add_argument("--save-report", type=str, default="", help="Optional path to save metrics JSON.")

    p.add_argument("--plot", action="store_true", help="Generate plots after writing report.json.")
    p.add_argument("--plot-only", action="store_true", help="Only plot from an existing report.json and exit.")
    p.add_argument("--report-path", type=str, default="", help="Optional report.json path for --plot-only.")
    p.add_argument("--plot-dir", type=str, default="", help="Optional output dir for plots (default: <out-dir>/plots).")
    p.add_argument("--plot-xscale", type=str, default="log", choices=["linear", "log"])
    p.add_argument("--plot-format", type=str, default="png", help="Plot format: png/pdf/svg (default: png).")

    p.add_argument("--write-example-config", type=str, default="", help="Write an example config JSON and exit.")
    args = p.parse_args(argv)

    if args.write_example_config:
        out = Path(args.write_example_config).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_example_config(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[msmacro_broad_test] wrote example config to {out}")
        return 0

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        report_path = (
            Path(args.report_path).expanduser().resolve()
            if args.report_path
            else out_dir / "report.json"
        )
        if not report_path.exists():
            raise SystemExit(f"report not found: {report_path}")
        report = _load_json(report_path)
        plot_dir = Path(args.plot_dir).expanduser().resolve() if args.plot_dir else None
        paths = plot_from_report(
            report=report,
            out_dir=out_dir,
            plot_dir=plot_dir,
            xscale=str(args.plot_xscale),
            fmt=str(args.plot_format),
        )
        for pth in paths:
            print(f"[msmacro_broad_test] wrote plot: {pth}")
        return 0

    if not args.config:
        raise SystemExit("--config is required (or use --write-example-config / --plot-only)")

    config_path = Path(args.config).expanduser().resolve()
    cfg_raw = _load_json(config_path)
    cfg = _validate_config(cfg_raw)

    _atomic_write_json(out_dir / "used_config.json", cfg_raw)

    collection_db_path = Path(args.collection_db).expanduser().resolve()
    queries_db_path = Path(args.queries_db).expanduser().resolve()
    qrels_path = Path(args.qrels).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()

    base, queries, qids, gt_pids, slice_info = _load_msmarco_unique_dev(
        collection_db_path=collection_db_path,
        queries_db_path=queries_db_path,
        qrels_path=qrels_path,
        cache_dir=cache_dir,
        max_db=args.max_db,
        max_queries=args.max_queries,
    )
    if slice_info["dropped_max_db"] > 0 and not args.allow_filtered_queries:
        raise SystemExit(
            "Refusing to run: --max-db would drop "
            f"{slice_info['dropped_max_db']} / {slice_info['queries_total']} queries "
            "(their gt_pid is outside the truncated database). "
            "Remove --max-db to run full MS MARCO (recommended), or pass --allow-filtered-queries "
            "if this subset evaluation is intentional."
        )
    n, d = int(base.shape[0]), int(base.shape[1])
    q = int(queries.shape[0])
    top_k = int(cfg["top_k"])
    if top_k > n:
        raise ValueError(f"top_k ({top_k}) must be <= database size ({n})")

    ks = _parse_int_list(args.report_ks)
    ks = [k for k in ks if 1 <= k <= top_k]
    if top_k not in ks:
        ks.append(top_k)
    ks = sorted(set(ks))

    _check_unit_norm_samples(base, name="base")
    _check_unit_norm_samples(queries, name="queries")

    print(
        "[msmacro_broad_test] "
        f"db={slice_info['db_used']}x{d} (total={slice_info['db_total']}), "
        f"queries={slice_info['queries_used']} (total={slice_info['queries_total']}, "
        f"dropped_max_db={slice_info['dropped_max_db']}, dropped_max_queries={slice_info['dropped_max_queries']}), "
        f"top_k={top_k}, metric={cfg['metric_name']}"
    )

    common_inputs = {
        "collection_db": _fingerprint(collection_db_path),
        "queries_db": _fingerprint(queries_db_path),
        "qrels": _fingerprint(qrels_path),
        "max_db": None if args.max_db is None else int(args.max_db),
        "max_queries": None if args.max_queries is None else int(args.max_queries),
        "metric": cfg["metric_name"],
        "slice": slice_info,
        "shapes": {"base": [n, d]},
    }

    report_rows: List[dict] = []

    for s in _iter_selected_series(cfg["series"], only=args.only_series):
        name = s["name"]
        kind = s["kind"]
        build = s["build"]
        sweep_key = s["sweep_key"]
        sweep_values = s["sweep_values"]

        series_inputs = {
            "name": name,
            "kind": kind,
            "build": build,
            "metric": cfg["metric_name"],
            "collection_db": common_inputs["collection_db"],
            "max_db": common_inputs["max_db"],
            "shape_base": [n, d],
        }
        series_id = f"{name}_{kind}_{_short_hash(series_inputs, n=10)}"

        expected_index_meta = dict(series_inputs)
        expected_index_meta.update({"faiss_index_type": kind})

        if args.dry_run:
            print(f"[dry-run] series={series_id} sweep {sweep_key}={sweep_values}")
            continue

        idx = _build_or_load_index(
            out_dir=out_dir,
            series_id=series_id,
            expected_meta=expected_index_meta,
            kind=kind,
            d=d,
            metric=cfg["metric"],
            build=build,
            base=base,
            batch_add=int(args.batch_add),
            threads=args.threads,
            force_rebuild=bool(args.force_rebuild_index),
        )

        for v in sweep_values:
            run_expected_meta = dict(common_inputs)
            run_expected_meta.update({"index_inputs": expected_index_meta})

            run_dir = _run_or_load_search(
                out_dir=out_dir,
                series_id=series_id,
                kind=kind,
                sweep_key=sweep_key,
                sweep_value=int(v),
                top_k=top_k,
                save_sims=bool(cfg["save_sims"]),
                index=idx,
                queries=queries,
                qids=qids,
                gt_pids=gt_pids,
                expected_meta=run_expected_meta,
                batch_queries=int(args.batch_queries),
                threads=args.threads,
                force_recompute=bool(args.force_search),
            )

            meta = _load_json(run_dir / "meta.json")
            rank = np.load(run_dir / "gt_rank.npy", mmap_mode="r")
            metrics = _metrics_from_rank(rank, ks=ks)
            qps_total = meta.get("timing_s", {}).get("qps_total")

            row = {
                "series_id": series_id,
                "name": name,
                "kind": kind,
                "sweep_key": sweep_key,
                "sweep_value": int(v),
                "qps_total": qps_total,
                "metrics": metrics,
                "run_dir": str(run_dir),
            }
            report_rows.append(row)

            # Print a compact line using common MS MARCO style: MRR@10 primarily.
            mrr10 = metrics.get(10, {}).get("mrr")
            hit10 = metrics.get(10, {}).get("hit")
            ndcg10 = metrics.get(10, {}).get("ndcg")
            qps_str = f"{float(qps_total):.2f}" if isinstance(qps_total, (int, float)) else "n/a"
            print(
                f"[{series_id}] {sweep_key}={int(v):4d} qps={qps_str:>7}  "
                f"hit@10={hit10:.6f} mrr@10={mrr10:.6f} ndcg@10={ndcg10:.6f}"
            )

    if args.dry_run:
        return 0

    report_path = (
        Path(args.save_report).expanduser().resolve()
        if args.save_report
        else out_dir / "report.json"
    )
    report_payload = {
        "created_at": _now_tag(),
        "slice": slice_info,
        "metric": cfg["metric_name"],
        "rows": report_rows,
        "ks": ks,
    }
    _atomic_write_json(
        report_path,
        report_payload,
    )
    print(f"[msmacro_broad_test] wrote report: {report_path}")

    if args.plot:
        plot_dir = Path(args.plot_dir).expanduser().resolve() if args.plot_dir else None
        paths = plot_from_report(
            report=report_payload,
            out_dir=out_dir,
            plot_dir=plot_dir,
            xscale=str(args.plot_xscale),
            fmt=str(args.plot_format),
        )
        for pth in paths:
            print(f"[msmacro_broad_test] wrote plot: {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
