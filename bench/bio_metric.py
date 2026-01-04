import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from ivf_pq import MAX_SCALE, rescale_database, rescale_query
from ivf_pq.merkle_zk import (
    _build_cluster_capacity,
    ivf_pq_learn as zk_ivf_pq_learn,
    zk_ivf_pq_query,
)
from ivf_pq.standard import ivf_pq_learn, ivf_pq_query
from vec_data_load.bupt_cbface_load import sample_bupt_cbface_queries_db_ground_truth


RESULT_DIR = Path("data") / "bio_metric"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = RESULT_DIR / "topk_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

METRIC_KEYS: Tuple[str, str] = ("standard_hit_at_k", "zk_hit_at_k")

CI95_KEYS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "standard": (
        ("hit_at_k", "standard_hit_at_k"),
        ("recall_at_k", "standard_recall_at_k"),
    ),
    "zk": (
        ("hit_at_k", "zk_hit_at_k"),
        ("recall_at_k", "zk_recall_at_k"),
        ("changed_count", "zk_changed_count"),
    ),
}


def _result_file_name(
    name: str,
    N: int,
    D: int,
    Q: int,
    K_gt: int,
    n_list: int,
    M: int,
    K: int,
    n_probe: int,
    top_k: int,
    scale_n: int,
    cluster_bound: int | None,
) -> Path:
    filename = (
        f"bio_{name}"
        f"_N{N}_D{D}_Q{Q}_Kgt{K_gt}"
        f"_nlist{n_list}_M{M}_K{K}"
        f"_nprobe{n_probe}_topk{top_k}_scale{scale_n}"
    )
    if cluster_bound is not None:
        filename += f"_cb{cluster_bound}"
    filename += ".json"
    return RESULT_DIR / filename


def _topk_cache_file_name(
    name: str,
    num_queries: int,
    ground_truth_k: int,
    n_list: int,
    M: int,
    K: int,
    n_probe: int,
    top_k: int,
    scale_n: int,
    cluster_bound: int | None,
    sample_seed: int,
    train_seed: int,
) -> Path:
    filename = (
        f"topk_{name}"
        f"_Q{num_queries}_Kgt{ground_truth_k}"
        f"_nlist{n_list}_M{M}_K{K}"
        f"_nprobe{n_probe}_topk{top_k}_scale{scale_n}"
        f"_sample{sample_seed}_train{train_seed}"
    )
    if cluster_bound is not None:
        filename += f"_cb{cluster_bound}"
    filename += ".npz"
    return CACHE_DIR / filename


def _load_cached(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    runs = payload.get("runs", [])
    out: List[Dict[str, float]] = []
    for run in runs:
        if all(k in run for k in METRIC_KEYS):
            cleaned: Dict[str, float] = {
                "standard_hit_at_k": float(run["standard_hit_at_k"]),
                "zk_hit_at_k": float(run["zk_hit_at_k"]),
            }
            optional_keys = [
                "zk_n",
                "zk_changed_count",
                "standard_train_time",
                "standard_query_time",
                "zk_train_time",
                "zk_query_time",
                "standard_recall_at_k",
                "zk_recall_at_k",
            ]
            for key in optional_keys:
                if key in run:
                    cleaned[key] = float(run[key])
            out.append(cleaned)
    return out


def _save_cached(
    path: Path,
    name: str,
    config: Dict[str, int],
    runs: List[Dict[str, float]],
) -> None:
    payload = {
        "name": name,
        "config": config,
        "metrics": list(METRIC_KEYS),
        "runs": runs,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _compute_summary(runs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not runs:
        raise ValueError("No runs provided for summary")

    summary: Dict[str, Dict[str, float]] = {}
    for group, metrics in CI95_KEYS.items():
        group_summary: Dict[str, float] = {}
        for metric_name, run_key in metrics:
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

            group_summary[f"mean_{metric_name}"] = mean
            group_summary[f"ci95_{metric_name}"] = ci95

        summary[group] = group_summary

    return summary


def _hit_curve(pred_topk: np.ndarray, gt_vecs: np.ndarray) -> np.ndarray:
    """
    pred_topk: (Q, K)
    gt_vecs: (Q, K_gt)
    return: hit@k curve, shape (K,), where out[k-1] = mean(hit@k)
    """
    pred = np.asarray(pred_topk)
    gt = np.asarray(gt_vecs)
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError("pred_topk and gt_vecs must be 2D arrays")
    if pred.shape[0] != gt.shape[0]:
        raise ValueError(f"Q mismatch: pred Q={pred.shape[0]} vs gt Q={gt.shape[0]}")

    Q, K = pred.shape
    hits = np.zeros((K,), dtype=np.float64)
    for i in range(Q):
        gt_set = set(int(x) for x in gt[i].tolist())
        found = False
        for k in range(K):
            if int(pred[i, k]) in gt_set:
                found = True
            if found:
                hits[k] += 1.0
    return hits / float(Q)


def _recall_curve(pred_topk: np.ndarray, gt_vecs: np.ndarray) -> np.ndarray:
    """
    pred_topk: (Q, K)
    gt_vecs: (Q, K_gt)
    return: recall@k curve, shape (K,), where out[k-1] = mean(recall@k)

    recall@k(query i) = |{pred[i, :k]} ∩ gt[i]| / |gt[i]|
    """
    pred = np.asarray(pred_topk)
    gt = np.asarray(gt_vecs)
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError("pred_topk and gt_vecs must be 2D arrays")
    if pred.shape[0] != gt.shape[0]:
        raise ValueError(f"Q mismatch: pred Q={pred.shape[0]} vs gt Q={gt.shape[0]}")

    Q, K = pred.shape
    recalls = np.zeros((K,), dtype=np.float64)
    for i in range(Q):
        gt_set = set(int(x) for x in gt[i].tolist())
        if not gt_set:
            raise ValueError(f"Empty ground truth for query {i}")
        denom = float(len(gt_set))

        seen: set[int] = set()
        hit_count = 0
        for k in range(K):
            idx = int(pred[i, k])
            if idx not in seen:
                seen.add(idx)
                if idx in gt_set:
                    hit_count += 1
            recalls[k] += hit_count / denom
    return recalls / float(Q)


def save_topk_cache(
    path: str | Path,
    *,
    gt: np.ndarray,
    standard_topk: np.ndarray,
    zk_topk: np.ndarray,
    meta: Dict[str, object] | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {} if meta is None else dict(meta)
    np.savez_compressed(
        str(p),
        gt=np.asarray(gt, dtype=np.int32),
        standard_topk=np.asarray(standard_topk, dtype=np.int32),
        zk_topk=np.asarray(zk_topk, dtype=np.int32),
        meta_json=np.array(json.dumps(payload), dtype=str),
    )
    return p


def load_topk_cache(
    path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    p = Path(path)
    data = np.load(str(p), allow_pickle=False)
    meta = json.loads(data["meta_json"].item())
    return data["gt"], data["standard_topk"], data["zk_topk"], meta


def hit_at_k(pred_topk: np.ndarray, gt_vecs: np.ndarray, k: int) -> float:
    pred = np.asarray(pred_topk)
    if pred.ndim != 2:
        raise ValueError("pred_topk must be a 2D array")
    if k <= 0 or k > pred.shape[1]:
        raise ValueError(f"k must be in [1, {pred.shape[1]}]")
    curve = _hit_curve(pred[:, :k], gt_vecs)
    return float(curve[-1])


def recall_at_k(pred_topk: np.ndarray, gt_vecs: np.ndarray, k: int) -> float:
    pred = np.asarray(pred_topk)
    if pred.ndim != 2:
        raise ValueError("pred_topk must be a 2D array")
    if k <= 0 or k > pred.shape[1]:
        raise ValueError(f"k must be in [1, {pred.shape[1]}]")
    curve = _recall_curve(pred[:, :k], gt_vecs)
    return float(curve[-1])


def hit_curve_from_cache(path: str | Path) -> Dict[str, np.ndarray]:
    gt, std_topk, zk_topk, _ = load_topk_cache(path)
    return {
        "standard_hit_curve": _hit_curve(std_topk, gt),
        "zk_hit_curve": _hit_curve(zk_topk, gt),
    }


def recall_curve_from_cache(path: str | Path) -> Dict[str, np.ndarray]:
    gt, std_topk, zk_topk, _ = load_topk_cache(path)
    return {
        "standard_recall_curve": _recall_curve(std_topk, gt),
        "zk_recall_curve": _recall_curve(zk_topk, gt),
    }


def _run_once_with_preds(
    database_vecs: np.ndarray,
    query_vecs: np.ndarray,
    gt_vecs: np.ndarray,
    top_k: int,
    n_list: int,
    M: int,
    K: int,
    n_probe: int,
    scale_n: int,
    cluster_bound: int | None,
    *,
    std_seed: int | None = None,
    zk_seed: int | None = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    进行一次训练 + 检索，对标准 IVF-PQ 与 ZK IVF-PQ
    分别计算 face-identification 指标（多正样本）：
      - hit@k：对每个 query，只要 top_k 里命中任意一个同人样本 (gt_vecs[i]) 就算成功。
      - recall@k：对每个 query，top_k 命中同人样本的比例（命中数 / |gt_vecs[i]|）。
    """
    base = np.asarray(database_vecs, dtype=np.float32)
    queries = np.asarray(query_vecs, dtype=np.float32)
    gt_arr = np.asarray(gt_vecs)

    if base.ndim != 2 or queries.ndim != 2:
        raise ValueError("database_vecs and query_vecs must be 2D arrays")
    if gt_arr.ndim != 2:
        raise ValueError("gt_vecs must be a 2D array of shape (Q, K_gt)")

    N, D = base.shape
    Q, Dq = queries.shape
    if D != Dq:
        raise ValueError(f"dimension mismatch: database D={D}, query D={Dq}")
    if gt_arr.shape[0] != Q:
        raise ValueError(f"gt_vecs Q mismatch: expected {Q}, got {gt_arr.shape[0]}")
    if top_k <= 0 or top_k > N:
        raise ValueError("top_k must be in [1, N]")

    if (gt_arr < 0).any() or (gt_arr >= N).any():
        raise ValueError("gt_vecs contains out-of-range indices for the provided database_vecs")

    if std_seed is None or zk_seed is None:
        rng = np.random.default_rng()
        if std_seed is None:
            std_seed = int(rng.integers(0, 2**31 - 1))
        if zk_seed is None:
            zk_seed = int(rng.integers(0, 2**31 - 1))

    # 1) 非 ZK 版本：使用浮点 standard IVF-PQ
    t0 = time.time()
    std_labels, std_center, std_code_books, std_quant_vecs, std_id_groups = ivf_pq_learn(
        base,
        n_list=n_list,
        M=M,
        K=K,
        random_state=std_seed,
    )
    standard_train_time = time.time() - t0
    print(f"[bio_metric] standard_train_time={standard_train_time:.3f}s")

    std_hit_list: List[float] = []
    std_recall_list: List[float] = []
    std_topk = np.empty((Q, top_k), dtype=np.int64)
    t0 = time.time()
    for i in tqdm(range(Q), "standard"):
        pred = ivf_pq_query(
            queries[i],
            top_k,
            std_labels,
            std_center,
            std_code_books,
            std_quant_vecs,
            std_id_groups,
            n_probe=n_probe,
        )
        pred_arr = np.asarray(pred, dtype=np.int64)
        if pred_arr.shape[0] != top_k:
            raise ValueError(f"standard returned top_k mismatch: got {pred_arr.shape[0]}, want {top_k}")
        std_topk[i] = pred_arr
        pred_set = set(int(x) for x in pred_arr.tolist())
        gt_set = set(int(x) for x in gt_arr[i].tolist())
        if not gt_set:
            raise ValueError(f"Empty ground truth for query {i}")
        inter = pred_set & gt_set
        std_hit_list.append(1.0 if inter else 0.0)
        std_recall_list.append(float(len(inter) / float(len(gt_set))))
    standard_query_time = time.time() - t0
    print(f"[bio_metric] standard_query_time={standard_query_time:.3f}s")

    # 2) ZK 版本：首先 rescale，然后使用 zk 版本的 learn + query
    t0 = time.time()
    scaled_base, v_min, v_max = rescale_database(base, scale_n)
    zk_changed_count: int | None = None
    if cluster_bound is not None:
        (
            zk_labels,
            zk_center,
            zk_code_books,
            zk_quant_vecs,
            zk_id_groups,
            zk_changed_count,
        ) = zk_ivf_pq_learn(
            scaled_base,
            n_list=n_list,
            M=M,
            K=K,
            random_state=zk_seed,
            cluster_bound=cluster_bound,
        )
    else:
        (
            zk_labels,
            zk_center,
            zk_code_books,
            zk_quant_vecs,
            zk_id_groups,
        ) = zk_ivf_pq_learn(
            scaled_base,
            n_list=n_list,
            M=M,
            K=K,
            random_state=zk_seed,
        )

    # 计算 ZK 证明中每簇需要 padding 到的容量 n（power-of-two 容量）
    zk_n = _build_cluster_capacity(zk_id_groups, n_probe)
    zk_train_time = time.time() - t0
    print(f"[bio_metric] zk_train_time={zk_train_time:.3f}s")

    zk_hit_list: List[float] = []
    zk_recall_list: List[float] = []
    zk_topk = np.empty((Q, top_k), dtype=np.int64)
    t0 = time.time()
    for i in tqdm(range(Q), "zk"):
        scaled_query = rescale_query(queries[i], scale_n, v_min, v_max)
        pred_zk, _ = zk_ivf_pq_query(
            scaled_query,
            zk_center,
            zk_code_books,
            zk_quant_vecs,
            zk_id_groups,
            top_k=top_k,
            n_probe=n_probe,
            proof=False,
        )
        pred_arr = np.asarray(pred_zk, dtype=np.int64)
        if pred_arr.shape[0] != top_k:
            raise ValueError(f"zk returned top_k mismatch: got {pred_arr.shape[0]}, want {top_k}")
        zk_topk[i] = pred_arr
        pred_set = set(int(x) for x in pred_arr.tolist())
        gt_set = set(int(x) for x in gt_arr[i].tolist())
        if not gt_set:
            raise ValueError(f"Empty ground truth for query {i}")
        inter = pred_set & gt_set
        zk_hit_list.append(1.0 if inter else 0.0)
        zk_recall_list.append(float(len(inter) / float(len(gt_set))))
    zk_query_time = time.time() - t0
    print(f"[bio_metric] zk_query_time={zk_query_time:.3f}s")

    result = {
        "standard_hit_at_k": float(np.mean(std_hit_list)),
        "standard_recall_at_k": float(np.mean(std_recall_list)),
        "zk_hit_at_k": float(np.mean(zk_hit_list)),
        "zk_recall_at_k": float(np.mean(zk_recall_list)),
        "standard_train_time": float(standard_train_time),
        "standard_query_time": float(standard_query_time),
        "zk_train_time": float(zk_train_time),
        "zk_query_time": float(zk_query_time),
        "zk_n": float(zk_n),
        "std_seed": int(std_seed),
        "zk_seed": int(zk_seed),
    }
    if zk_changed_count is not None:
        result["zk_changed_count"] = float(zk_changed_count)
    return result, std_topk, zk_topk


def _run_once(
    database_vecs: np.ndarray,
    query_vecs: np.ndarray,
    gt_vecs: np.ndarray,
    top_k: int,
    n_list: int,
    M: int,
    K: int,
    n_probe: int,
    scale_n: int,
    cluster_bound: int | None,
) -> Dict[str, float]:
    result, _, _ = _run_once_with_preds(
        database_vecs,
        query_vecs,
        gt_vecs,
        top_k,
        n_list,
        M,
        K,
        n_probe,
        scale_n,
        cluster_bound,
    )
    return result


def run_bio_metric_bench(
    database_vecs: np.ndarray,
    query_vecs: np.ndarray,
    gt_vecs: np.ndarray,
    top_k: int,
    name: str,
    *,
    n_list: int = 64,
    M: int = 8,
    K: int = 256,
    n_probe: int = 8,
    scale_n: int = MAX_SCALE,
    cluster_bound: int | None = None,
    num_runs: int = 5,
    force_recompute: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    在 face-identification ground truth（同人集合）下比较 standard IVF-PQ 与 ZK IVF-PQ 的 hit@k。
    """
    base = np.asarray(database_vecs)
    queries = np.asarray(query_vecs)
    gt_arr = np.asarray(gt_vecs)

    if base.ndim != 2 or queries.ndim != 2:
        raise ValueError("database_vecs and query_vecs must be 2D arrays")

    N, D = base.shape
    Q, Dq = queries.shape
    if D != Dq:
        raise ValueError(f"dimension mismatch: database D={D}, query D={Dq}")

    if gt_arr.ndim != 2:
        raise ValueError("gt_vecs must be a 2D array of shape (Q, K_gt)")
    Q_gt, K_gt = gt_arr.shape
    if Q_gt != Q:
        raise ValueError(f"gt_vecs Q mismatch: expected {Q}, got {Q_gt}")

    path = _result_file_name(
        name=name,
        N=N,
        D=D,
        Q=Q,
        K_gt=K_gt,
        n_list=n_list,
        M=M,
        K=K,
        n_probe=n_probe,
        top_k=top_k,
        scale_n=scale_n,
        cluster_bound=cluster_bound,
    )

    if force_recompute:
        runs: List[Dict[str, float]] = []
    else:
        runs = _load_cached(path)

    existing_runs = len(runs)
    if existing_runs < num_runs:
        for _ in range(num_runs - existing_runs):
            runs.append(
                _run_once(
                    base,
                    queries,
                    gt_arr,
                    top_k,
                    n_list,
                    M,
                    K,
                    n_probe,
                    scale_n,
                    cluster_bound,
                )
            )
        config = {
            "N": N,
            "D": D,
            "Q": Q,
            "K_gt": K_gt,
            "n_list": n_list,
            "M": M,
            "K": K,
            "n_probe": n_probe,
            "top_k": top_k,
            "scale_n": scale_n,
            "cluster_bound": cluster_bound,
        }
        _save_cached(path, name, config, runs)

    return _compute_summary(runs)


def run_bupt_cbface_bench(
    dataset_dir: str | Path,
    *,
    top_k: int = 100,
    name: str | None = None,
    num_queries: int = 1024,
    ground_truth_k: int = 11,
    sample_seed: int = 0,
    db_name: str = "arcface_embeddings.duckdb",
    n_list: int = 1024,
    M: int = 8,
    K: int = 256,
    n_probe: int = 8,
    scale_n: int = MAX_SCALE,
    cluster_bound: int | None = 2048,
    num_runs: int = 5,
    force_recompute: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    直接从 `dataset_dir` 下的 DuckDB 采样 (queries, db, gt)，并跑 hit@k bench。
    """
    if name is None:
        dataset_name = Path(dataset_dir).name
        name = f"{dataset_name}_seed{sample_seed}"

    queries, db, gt = sample_bupt_cbface_queries_db_ground_truth(
        dataset_dir,
        db_name=db_name,
        num_queries=num_queries,
        ground_truth_k=ground_truth_k,
        seed=sample_seed,
    )
    return run_bio_metric_bench(
        db,
        queries,
        gt,
        top_k=top_k,
        name=name,
        n_list=n_list,
        M=M,
        K=K,
        n_probe=n_probe,
        scale_n=scale_n,
        cluster_bound=cluster_bound,
        num_runs=num_runs,
        force_recompute=force_recompute,
    )


def cache_bupt_cbface_topk(
    dataset_dir: str | Path,
    *,
    top_k: int = 100,
    name: str | None = None,
    num_queries: int = 1024,
    ground_truth_k: int = 11,
    sample_seed: int = 0,
    train_seed: int = 0,
    db_name: str = "arcface_embeddings.duckdb",
    n_list: int = 1024,
    M: int = 8,
    K: int = 256,
    n_probe: int = 8,
    scale_n: int = MAX_SCALE,
    cluster_bound: int | None = 2048,
    cache_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """
    生成并缓存一次 top-k 检索结果（standard + zk）以及 ground-truth。

    产物是一个 `.npz` 文件，包含：
      - gt: (Q, K_gt)
      - standard_topk: (Q, top_k)
      - zk_topk: (Q, top_k)
      - meta_json: 方便记录参数/时间（非强约束）
    """
    if name is None:
        dataset_name = Path(dataset_dir).name
        name = f"{dataset_name}_seed{sample_seed}"

    if cache_path is None:
        cache_path = _topk_cache_file_name(
            name=name,
            num_queries=num_queries,
            ground_truth_k=ground_truth_k,
            n_list=n_list,
            M=M,
            K=K,
            n_probe=n_probe,
            top_k=top_k,
            scale_n=scale_n,
            cluster_bound=cluster_bound,
            sample_seed=sample_seed,
            train_seed=train_seed,
        )

    cache_path = Path(cache_path)
    if cache_path.exists() and not overwrite:
        return cache_path

    seed_rng = np.random.default_rng(train_seed)
    std_seed = int(seed_rng.integers(0, 2**31 - 1))
    zk_seed = int(seed_rng.integers(0, 2**31 - 1))

    queries, db, gt = sample_bupt_cbface_queries_db_ground_truth(
        dataset_dir,
        db_name=db_name,
        num_queries=num_queries,
        ground_truth_k=ground_truth_k,
        seed=sample_seed,
    )

    run, std_topk, zk_topk = _run_once_with_preds(
        database_vecs=db,
        query_vecs=queries,
        gt_vecs=gt,
        top_k=top_k,
        n_list=n_list,
        M=M,
        K=K,
        n_probe=n_probe,
        scale_n=scale_n,
        cluster_bound=cluster_bound,
        std_seed=std_seed,
        zk_seed=zk_seed,
    )

    meta: Dict[str, object] = {
        "name": name,
        "sample_seed": int(sample_seed),
        "train_seed": int(train_seed),
        "ground_truth_k": int(ground_truth_k),
        "db_name": str(db_name),
        "n_list": int(n_list),
        "M": int(M),
        "K": int(K),
        "n_probe": int(n_probe),
        "top_k": int(top_k),
        "scale_n": int(scale_n),
        "cluster_bound": None if cluster_bound is None else int(cluster_bound),
        "run": run,
    }

    return save_topk_cache(
        cache_path,
        gt=gt,
        standard_topk=std_topk,
        zk_topk=zk_topk,
        meta=meta,
    )
