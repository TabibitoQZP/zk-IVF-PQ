import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss
import numpy as np

from ivf_pq.layout import apply_layout, layout_suffix, normalize_layout
from ivf_pq.util.fread import read_fvecs
from vec_data_load.sift import SIFT


RESULT_DIR = Path("data") / "faiss_opq_bench"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_REPORT_KS: Tuple[int, ...] = (1, 10, 50, 100)
METRIC_VERSION = 1


def _parse_report_ks(arg: str | None) -> list[int] | None:
    if arg is None:
        return None
    raw = str(arg).strip()
    if not raw:
        return None

    values: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    return values or None


def _resolve_layout(data_name: str, layout_arg: str | None) -> str | None:
    token = "auto" if layout_arg is None else str(layout_arg).strip().lower()
    if token in ("", "auto"):
        if data_name in {"sift", "gist"}:
            return "mod8"
        return None
    if token == "none":
        return None
    if token == "mod8":
        return token
    raise ValueError(f"Unsupported layout: {layout_arg!r}")


def _normalize_report_ks(
    top_k: int,
    report_ks: Iterable[int] | None = None,
) -> Tuple[int, ...]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    raw_ks = DEFAULT_REPORT_KS if report_ks is None else tuple(int(k) for k in report_ks)
    normalized = {int(top_k)}
    for k in raw_ks:
        if k <= 0:
            raise ValueError(f"report_ks must contain positive integers, got {k}")
        if k <= top_k:
            normalized.add(int(k))

    out = tuple(sorted(normalized))
    if not out:
        raise ValueError("report_ks must contain at least one valid k")
    return out


def _metric_key(metric: str, k: int) -> str:
    return f"{metric}_at_{int(k)}"


def _query_metrics(
    *,
    pred: np.ndarray,
    gt_topk: np.ndarray,
    report_ks: Tuple[int, ...],
) -> Dict[str, float]:
    pred_arr = np.asarray(pred, dtype=np.int64).reshape(-1)
    gt_arr = np.asarray(gt_topk, dtype=np.int64).reshape(-1)
    if gt_arr.size == 0:
        raise ValueError("ground-truth top-k must be non-empty")

    best_gt = int(gt_arr[0])
    out: Dict[str, float] = {}
    for k in report_ks:
        pred_prefix = pred_arr[: min(int(k), pred_arr.size)]
        gt_prefix = gt_arr[: int(k)]
        inter = np.intersect1d(pred_prefix, gt_prefix)
        out[_metric_key("pass", k)] = float(inter.size) / float(k)
        out[_metric_key("recall", k)] = 1.0 if best_gt in pred_prefix else 0.0
    return out


def _append_metric_lists(
    metric_lists: Dict[str, List[float]],
    metric_values: Dict[str, float],
) -> None:
    for key, value in metric_values.items():
        metric_lists[key].append(float(value))


def _mean_metric_lists(metric_lists: Dict[str, List[float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, values in metric_lists.items():
        if not values:
            continue
        out[key] = float(np.mean(values))
    return out


def _power_of_two_nbits(K: int) -> int:
    if K <= 0:
        raise ValueError("K must be positive")
    nbits = int(round(math.log2(int(K))))
    if (1 << nbits) != int(K):
        raise ValueError(f"K must be a power of two for FAISS PQ, got K={K}")
    return int(nbits)


def _sample_train_vectors(
    train_pool: np.ndarray,
    train_size: int | None,
    random_state: int | None,
) -> tuple[np.ndarray, int]:
    pool = np.asarray(train_pool, dtype=np.float32)
    if pool.ndim != 2:
        raise ValueError("train_pool must be 2D")

    if train_size is None:
        return np.ascontiguousarray(pool), int(pool.shape[0])

    if train_size <= 0:
        raise ValueError("train_size must be positive when provided")

    actual = min(int(train_size), int(pool.shape[0]))
    if actual == pool.shape[0]:
        return np.ascontiguousarray(pool), actual

    rng = np.random.default_rng(seed=random_state)
    train_idx = rng.choice(pool.shape[0], size=actual, replace=False)
    return np.ascontiguousarray(pool[train_idx]), actual


def _result_file_name(
    *,
    name: str,
    N: int,
    D: int,
    Q: int,
    n_list: int,
    M: int,
    K: int,
    n_probe: int,
    top_k: int,
    train_count: int,
    use_gpu: bool,
    layout: str | None,
) -> Path:
    filename = (
        f"faiss_opq_{name}"
        f"_N{N}_D{D}_Q{Q}"
        f"_nlist{n_list}_M{M}_K{K}"
        f"_nprobe{n_probe}_topk{top_k}"
        f"_train{train_count}_gpu{1 if use_gpu else 0}"
    )
    filename += layout_suffix(layout)
    filename += ".json"
    return RESULT_DIR / filename


def _required_run_keys(report_ks: Tuple[int, ...]) -> Tuple[str, ...]:
    keys = ["train_time", "add_time", "query_time", "train_count", "used_gpu"]
    for metric in ("pass", "recall"):
        for k in report_ks:
            keys.append(_metric_key(metric, k))
    return tuple(keys)


def _config_matches(payload_config: dict, expected_config: dict) -> bool:
    keys = (
        "N",
        "D",
        "Q",
        "n_list",
        "M",
        "K",
        "n_probe",
        "top_k",
        "report_ks",
        "layout",
        "use_gpu",
        "gpu_device",
        "train_size",
        "train_count",
        "opq_niter",
        "precompute_lut",
    )
    for key in keys:
        if payload_config.get(key) != expected_config.get(key):
            return False
    return True


def _load_cached(
    path: Path,
    *,
    expected_config: dict,
    report_ks: Tuple[int, ...],
) -> List[Dict[str, float]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    config = payload.get("config", {})
    if not isinstance(config, dict) or not _config_matches(config, expected_config):
        return []

    required_keys = set(_required_run_keys(report_ks))
    out: List[Dict[str, float]] = []
    for run in payload.get("runs", []):
        if not required_keys.issubset(run.keys()):
            continue
        cleaned: Dict[str, float] = {}
        for key, value in run.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                cleaned[key] = float(value)
        out.append(cleaned)
    return out


def _save_cached(
    path: Path,
    *,
    name: str,
    config: dict,
    runs: List[Dict[str, float]],
    summary: dict | None,
) -> None:
    payload = {
        "name": name,
        "metric_version": METRIC_VERSION,
        "config": config,
        "metrics": sorted(runs[0].keys()) if runs else [],
        "runs": runs,
    }
    if summary is not None:
        payload["summary"] = summary

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _compute_summary(
    runs: List[Dict[str, float]],
    *,
    top_k: int,
    report_ks: Tuple[int, ...],
) -> Dict[str, Dict[str, float]]:
    if not runs:
        raise ValueError("No runs provided for summary")

    block: Dict[str, float] = {}
    for metric in ("pass", "recall"):
        for k in report_ks:
            run_key = _metric_key(metric, k)
            values = [float(run[run_key]) for run in runs if run_key in run]
            if not values:
                continue

            arr = np.asarray(values, dtype=float)
            mean = float(arr.mean())
            if arr.size > 1:
                std = float(arr.std(ddof=1))
                ci95 = float(1.96 * std / math.sqrt(int(arr.size)))
            else:
                ci95 = 0.0

            block[f"mean_{metric}_at_{k}"] = mean
            block[f"ci95_{metric}_at_{k}"] = ci95

    for metric_name in ("train_time", "add_time", "query_time", "train_count", "used_gpu"):
        values = [float(run[metric_name]) for run in runs if metric_name in run]
        if not values:
            continue

        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        if arr.size > 1:
            std = float(arr.std(ddof=1))
            ci95 = float(1.96 * std / math.sqrt(int(arr.size)))
        else:
            ci95 = 0.0

        block[f"mean_{metric_name}"] = mean
        block[f"ci95_{metric_name}"] = ci95

    alias_mean_key = f"mean_pass_at_{top_k}"
    alias_ci_key = f"ci95_pass_at_{top_k}"
    if alias_mean_key in block:
        block["mean_pass_at_k"] = block[alias_mean_key]
    if alias_ci_key in block:
        block["ci95_pass_at_k"] = block[alias_ci_key]
        block["ci95"] = block[alias_ci_key]

    alias_mean_key = f"mean_recall_at_{top_k}"
    alias_ci_key = f"ci95_recall_at_{top_k}"
    if alias_mean_key in block:
        block["mean_recall_at_k"] = block[alias_mean_key]
    if alias_ci_key in block:
        block["ci95_recall_at_k"] = block[alias_ci_key]

    return {"faiss_opq": block}


def _load_dataset(
    data_name: str,
    data_root: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    data_dir = Path(data_root) / data_name
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {data_dir}")

    dataset = SIFT(str(data_dir))
    prefix = data_dir.name
    learn_path = data_dir / f"{prefix}_learn.fvecs"
    learn_vecs: np.ndarray | None = None
    if learn_path.exists():
        learn_vecs = read_fvecs(str(learn_path))

    return dataset.base_vecs, dataset.query_vecs, dataset.gt_vecs, learn_vecs


def _maybe_to_gpu(
    index_cpu,
    *,
    use_gpu: bool,
    gpu_device: int,
):
    if not use_gpu:
        return index_cpu, False, None

    if not hasattr(faiss, "StandardGpuResources"):
        print("[faiss_opq_bench] faiss-gpu not detected, using CPU.")
        return index_cpu, False, None

    try:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, int(gpu_device), index_cpu)
        return index, True, res
    except Exception as exc:
        print(f"[faiss_opq_bench] GPU unavailable, fallback to CPU: {exc}")
        return index_cpu, False, None


def _set_nprobe(index, n_probe: int) -> None:
    target = int(n_probe)
    errors: list[str] = []

    try:
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", target)
        return
    except Exception as exc:
        errors.append(f"ParameterSpace: {exc}")

    if hasattr(faiss, "GpuParameterSpace"):
        try:
            gpu_ps = faiss.GpuParameterSpace()
            try:
                gpu_ps.initialize(index)
            except Exception:
                # Some FAISS builds do not require/allow explicit initialize.
                pass
            gpu_ps.set_index_parameter(index, "nprobe", target)
            return
        except Exception as exc:
            errors.append(f"GpuParameterSpace: {exc}")

    try:
        faiss.extract_index_ivf(index).nprobe = target
        return
    except Exception as exc:
        errors.append(f"extract_index_ivf: {exc}")

    if hasattr(index, "nprobe"):
        try:
            index.nprobe = target
            return
        except Exception as exc:
            errors.append(f"index.nprobe: {exc}")

    raise RuntimeError(
        "Unable to set nprobe on FAISS index. "
        f"Attempted CPU/GPU parameter APIs and direct assignment: {' | '.join(errors)}"
    )


def _run_once(
    database_vecs: np.ndarray,
    query_vecs: np.ndarray,
    gt_vecs: np.ndarray,
    learn_vecs: np.ndarray | None,
    *,
    top_k: int,
    n_list: int,
    M: int,
    K: int,
    n_probe: int,
    train_size: int | None,
    layout: str | None,
    use_gpu: bool,
    gpu_device: int,
    precompute_lut: bool,
    opq_niter: int,
    random_state: int | None,
    report_ks: Tuple[int, ...],
) -> Dict[str, float]:
    base = apply_layout(np.asarray(database_vecs, dtype=np.float32), layout)
    queries = apply_layout(np.asarray(query_vecs, dtype=np.float32), layout)
    gt_arr = np.asarray(gt_vecs, dtype=np.int64)

    if base.ndim != 2 or queries.ndim != 2:
        raise ValueError("database_vecs and query_vecs must be 2D arrays")
    if gt_arr.ndim != 2:
        raise ValueError("gt_vecs must be a 2D array")

    N, D = base.shape
    Q, Dq = queries.shape
    if D != Dq:
        raise ValueError(f"dimension mismatch: database D={D}, query D={Dq}")
    if gt_arr.shape[0] != Q:
        raise ValueError(f"gt_vecs Q mismatch: expected {Q}, got {gt_arr.shape[0]}")
    if top_k > gt_arr.shape[1]:
        raise ValueError(
            f"top_k={top_k} exceeds available ground-truth size K_gt={gt_arr.shape[1]}"
        )
    if D % M != 0:
        raise ValueError(f"M={M} must divide D={D} exactly")

    nbits = _power_of_two_nbits(K)
    train_pool = base if learn_vecs is None else apply_layout(learn_vecs, layout)
    xt, actual_train_count = _sample_train_vectors(
        train_pool,
        train_size=train_size,
        random_state=random_state,
    )
    minimum_train_count = max(int(n_list), int(K))
    if actual_train_count < minimum_train_count:
        raise ValueError(
            "Not enough training vectors for FAISS OPQ/IVF-PQ: "
            f"need at least max(n_list={n_list}, K={K})={minimum_train_count}, "
            f"got {actual_train_count}"
        )

    quantizer = faiss.IndexFlatL2(D)
    ivfpq = faiss.IndexIVFPQ(quantizer, D, int(n_list), int(M), int(nbits), faiss.METRIC_L2)
    ivfpq.use_precomputed_table = bool(precompute_lut)

    opq = faiss.OPQMatrix(D, int(M))
    opq.pq = faiss.ProductQuantizer(D, int(M), int(nbits))
    opq.niter = int(opq_niter)
    index_cpu = faiss.IndexPreTransform(opq, ivfpq)

    index, used_gpu, _gpu_resources = _maybe_to_gpu(
        index_cpu,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
    )

    t0 = time.time()
    index.train(xt)
    train_time = time.time() - t0

    t0 = time.time()
    index.add(base)
    add_time = time.time() - t0

    _set_nprobe(index, int(n_probe))

    t0 = time.time()
    _, pred = index.search(queries, int(top_k))
    query_time = time.time() - t0

    gt_topk = gt_arr[:, :top_k]
    metric_lists = {
        _metric_key(metric, k): []
        for metric in ("pass", "recall")
        for k in report_ks
    }
    for i in range(Q):
        _append_metric_lists(
            metric_lists,
            _query_metrics(
                pred=pred[i],
                gt_topk=gt_topk[i],
                report_ks=report_ks,
            ),
        )

    result = {
        "train_time": float(train_time),
        "add_time": float(add_time),
        "query_time": float(query_time),
        "train_count": float(actual_train_count),
        "used_gpu": float(1.0 if used_gpu else 0.0),
    }
    result.update(_mean_metric_lists(metric_lists))
    return result


def run_faiss_opq_bench(
    database_vecs: np.ndarray,
    query_vecs: np.ndarray,
    gt_vecs: np.ndarray,
    learn_vecs: np.ndarray | None,
    top_k: int,
    name: str,
    *,
    n_list: int = 1024,
    M: int = 8,
    K: int = 256,
    n_probe: int = 8,
    num_runs: int = 1,
    train_size: int | None = None,
    layout: str | None = None,
    use_gpu: bool = False,
    gpu_device: int = 0,
    force_recompute: bool = False,
    report_ks: Iterable[int] | None = None,
    precompute_lut: bool = True,
    opq_niter: int = 20,
) -> Dict[str, Dict[str, float]]:
    base = np.asarray(database_vecs)
    queries = np.asarray(query_vecs)
    gt_arr = np.asarray(gt_vecs)
    if base.ndim != 2 or queries.ndim != 2:
        raise ValueError("database_vecs and query_vecs must be 2D arrays")
    if gt_arr.ndim != 2:
        raise ValueError("gt_vecs must be a 2D array")

    N, D = base.shape
    Q, Dq = queries.shape
    if D != Dq:
        raise ValueError(f"dimension mismatch: database D={D}, query D={Dq}")
    if gt_arr.shape[0] != Q:
        raise ValueError(f"gt_vecs Q mismatch: expected {Q}, got {gt_arr.shape[0]}")
    if top_k > gt_arr.shape[1]:
        raise ValueError(
            f"top_k={top_k} exceeds available ground-truth size K_gt={gt_arr.shape[1]}"
        )

    resolved_report_ks = _normalize_report_ks(top_k, report_ks)
    resolved_layout = normalize_layout(layout)

    train_pool = base if learn_vecs is None else np.asarray(learn_vecs)
    if train_pool.ndim != 2 or train_pool.shape[1] != D:
        raise ValueError("learn_vecs must be 2D and match database dimension")
    expected_train_count = int(
        train_pool.shape[0] if train_size is None else min(int(train_size), int(train_pool.shape[0]))
    )

    path = _result_file_name(
        name=name,
        N=N,
        D=D,
        Q=Q,
        n_list=n_list,
        M=M,
        K=K,
        n_probe=n_probe,
        top_k=top_k,
        train_count=expected_train_count,
        use_gpu=use_gpu,
        layout=resolved_layout,
    )

    config = {
        "N": N,
        "D": D,
        "Q": Q,
        "n_list": int(n_list),
        "M": int(M),
        "K": int(K),
        "n_probe": int(n_probe),
        "top_k": int(top_k),
        "report_ks": [int(k) for k in resolved_report_ks],
        "layout": resolved_layout,
        "use_gpu": bool(use_gpu),
        "gpu_device": int(gpu_device),
        "train_size": None if train_size is None else int(train_size),
        "train_count": int(expected_train_count),
        "opq_niter": int(opq_niter),
        "precompute_lut": bool(precompute_lut),
    }

    if force_recompute:
        runs: List[Dict[str, float]] = []
    else:
        runs = _load_cached(
            path,
            expected_config=config,
            report_ks=resolved_report_ks,
        )
        if path.exists() and not runs:
            print(f"[faiss_opq_bench] cache ignored: {path}")

    existing_runs = len(runs)
    if existing_runs < num_runs:
        seed_rng = np.random.default_rng()
        for _ in range(num_runs - existing_runs):
            runs.append(
                _run_once(
                    base,
                    queries,
                    gt_arr,
                    learn_vecs,
                    top_k=top_k,
                    n_list=n_list,
                    M=M,
                    K=K,
                    n_probe=n_probe,
                    train_size=train_size,
                    layout=resolved_layout,
                    use_gpu=use_gpu,
                    gpu_device=gpu_device,
                    precompute_lut=precompute_lut,
                    opq_niter=opq_niter,
                    random_state=int(seed_rng.integers(0, 2**31 - 1)),
                    report_ks=resolved_report_ks,
                )
            )
            _save_cached(path, name=name, config=config, runs=runs, summary=None)

    summary = _compute_summary(runs, top_k=top_k, report_ks=resolved_report_ks)
    _save_cached(path, name=name, config=config, runs=runs, summary=summary)
    return summary


def _print_metric_block(summary: dict, *, report_ks: list[int]) -> None:
    block = summary.get("faiss_opq", {})
    if not isinstance(block, dict):
        return

    print("[faiss_opq_bench] faiss_opq")
    for k in report_ks:
        pass_key = f"mean_pass_at_{k}"
        recall_key = f"mean_recall_at_{k}"
        if pass_key not in block and recall_key not in block:
            continue

        pass_val = block.get(pass_key)
        recall_val = block.get(recall_key)
        pass_ci = block.get(f"ci95_pass_at_{k}")
        recall_ci = block.get(f"ci95_recall_at_{k}")

        pass_txt = (
            f"{float(pass_val):.4f}+-{float(pass_ci):.4f}"
            if pass_val is not None and pass_ci is not None
            else "n/a"
        )
        recall_txt = (
            f"{float(recall_val):.4f}+-{float(recall_ci):.4f}"
            if recall_val is not None and recall_ci is not None
            else "n/a"
        )
        print(f"  @ {k:>3d}: pass={pass_txt}  recall={recall_txt}")

    for metric in ("train_time", "add_time", "query_time", "train_count", "used_gpu"):
        mean_key = f"mean_{metric}"
        if mean_key not in block:
            continue
        ci_key = f"ci95_{metric}"
        value = float(block[mean_key])
        ci = float(block.get(ci_key, 0.0))
        print(f"  {metric}: {value:.4f}+-{ci:.4f}")


def main(
    data_name: str,
    n_list: int,
    n_probe: int,
    *,
    data_root: str = "data",
    M: int = 8,
    K: int = 256,
    top_k: int = 100,
    num_runs: int = 1,
    train_size: int | None = None,
    force_recompute: bool = False,
    name: str | None = None,
    report_ks: list[int] | None = None,
    layout: str | None = "auto",
    use_gpu: bool = False,
    gpu_device: int = 0,
    precompute_lut: bool = True,
    opq_niter: int = 20,
) -> None:
    stime = time.time()
    base_vecs, query_vecs, gt_vecs, learn_vecs = _load_dataset(data_name, data_root)

    print(data_name)
    print(base_vecs.shape)
    if learn_vecs is not None:
        print(learn_vecs.shape)
    print(query_vecs.shape)
    print(gt_vecs.shape)

    if name is None:
        name = f"faiss_opq_{data_name}"

    resolved_layout = _resolve_layout(data_name, layout)
    print(f"[faiss_opq_bench] layout={resolved_layout or 'none'}")

    if learn_vecs is None:
        print("[faiss_opq_bench] learn set not found, fallback to sampling from base vectors.")

    summary = run_faiss_opq_bench(
        base_vecs,
        query_vecs,
        gt_vecs,
        learn_vecs,
        top_k=top_k,
        name=name,
        n_list=n_list,
        M=M,
        K=K,
        n_probe=n_probe,
        num_runs=num_runs,
        train_size=train_size,
        layout=resolved_layout,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        force_recompute=force_recompute,
        report_ks=report_ks,
        precompute_lut=precompute_lut,
        opq_niter=opq_niter,
    )
    print(summary)

    raw_ks = report_ks if report_ks is not None else list(DEFAULT_REPORT_KS)
    resolved_report_ks = sorted(
        {int(k) for k in raw_ks if int(k) <= int(top_k)} | {int(top_k)}
    )
    _print_metric_block(summary, report_ks=resolved_report_ks)
    print((time.time() - stime) / 3600)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "FAISS OPQ + IVF-PQ accuracy bench for SIFT/GIST. "
            "Uses learn/base/query/groundtruth when available, reports pass@k "
            "and recall@k for multiple ks, and caches results under "
            "data/faiss_opq_bench/."
        )
    )
    parser.add_argument(
        "--data-name",
        type=str,
        required=True,
        choices=["sift", "gist"],
        help="Dataset folder name under --data-root.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory that contains the dataset folders (default: data).",
    )
    parser.add_argument("--n-list", type=int, default=1024)
    parser.add_argument("--n-probe", type=int, default=8)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument(
        "--K",
        type=int,
        default=256,
        help="Number of PQ codewords per subspace. Must be a power of two (default: 256).",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--report-ks",
        type=str,
        default="",
        help=(
            "Comma-separated ks to report from the single top-k retrieval "
            f"(default: {','.join(str(k) for k in DEFAULT_REPORT_KS)}; top-k is always included)."
        ),
    )
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Number of learn vectors used for training. Default: use the full learn set.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Cache name prefix (default: faiss_opq_{data-name}).",
    )
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--layout",
        type=str,
        default="auto",
        choices=["auto", "none", "mod8"],
        help=(
            "Vector layout before OPQ/PQ. 'auto' defaults to mod8 for sift/gist, "
            "'none' keeps the raw dimension order."
        ),
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Try to move the FAISS index to GPU. Falls back to CPU on failure.",
    )
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument(
        "--opq-niter",
        type=int,
        default=20,
        help="Number of OPQ training iterations (default: 20).",
    )
    parser.add_argument(
        "--no-precompute-lut",
        action="store_true",
        help="Disable FAISS IVFPQ precomputed lookup tables.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(
        args.data_name,
        n_list=int(args.n_list),
        n_probe=int(args.n_probe),
        data_root=str(args.data_root),
        M=int(args.M),
        K=int(args.K),
        top_k=int(args.top_k),
        num_runs=int(args.num_runs),
        train_size=args.train_size,
        force_recompute=bool(args.force_recompute),
        name=args.name,
        report_ks=_parse_report_ks(args.report_ks),
        layout=str(args.layout),
        use_gpu=bool(args.use_gpu),
        gpu_device=int(args.gpu_device),
        precompute_lut=not bool(args.no_precompute_lut),
        opq_niter=int(args.opq_niter),
    )
