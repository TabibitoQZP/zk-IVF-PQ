import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from tqdm import tqdm

import numpy as np

from ivf_pq import MAX_SCALE, brute_force_knn, rescale_database, rescale_query
from ivf_pq.layout import layout_suffix, normalize_layout
from ivf_pq.merkle_zk import (
    ivf_pq_learn as zk_ivf_pq_learn,
    zk_ivf_pq_query,
    _build_cluster_capacity,
)
from ivf_pq.standard import ivf_pq_learn, ivf_pq_query


RESULT_DIR = Path("data") / "acc_bench"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_REPORT_KS: Tuple[int, ...] = (1, 10, 50, 100)
METRIC_VERSION = 3
SUMMARY_EXTRA_KEYS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "standard": (
        ("train_time", "standard_train_time"),
        ("query_time", "standard_query_time"),
    ),
    "zk": (
        ("train_time", "zk_train_time"),
        ("query_time", "zk_query_time"),
        ("changed_count", "zk_changed_count"),
    ),
    "shared": (("bruteforce_time", "bruteforce_time"),),
}


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
        normalized.add(int(k))

    out = tuple(sorted(normalized))
    if not out:
        raise ValueError("report_ks must contain at least one valid k")
    return out


def _metric_key(scheme: str, metric: str, k: int) -> str:
    return f"{scheme}_{metric}_at_{int(k)}"


def _legacy_alias_keys() -> Tuple[str, ...]:
    return (
        "standard_pass_at_k",
        "zk_pass_at_k",
        "standard_recall_at_k",
        "zk_recall_at_k",
    )


def _required_run_keys(report_ks: Tuple[int, ...]) -> Tuple[str, ...]:
    keys: List[str] = list(_legacy_alias_keys())
    for scheme in ("standard", "zk"):
        for metric in ("pass", "recall"):
            for k in report_ks:
                keys.append(_metric_key(scheme, metric, k))
    return tuple(keys)


def _add_legacy_aliases(result: Dict[str, float], top_k: int) -> None:
    result["standard_pass_at_k"] = float(result[_metric_key("standard", "pass", top_k)])
    result["zk_pass_at_k"] = float(result[_metric_key("zk", "pass", top_k)])
    result["standard_recall_at_k"] = float(
        result[_metric_key("standard", "recall", top_k)]
    )
    result["zk_recall_at_k"] = float(result[_metric_key("zk", "recall", top_k)])


def _query_metrics(
    *,
    scheme: str,
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
        # recall@k: ground truth 的前 k 个中，至少有一个被检索到
        inter = np.intersect1d(pred_prefix, gt_prefix)
        out[_metric_key(scheme, "recall", k)] = 1.0 if inter.size > 0 else 0.0
        # pass@k: 仅当 k <= 实际检索数量时才计算
        if k <= pred_arr.size:
            out[_metric_key(scheme, "pass", k)] = float(inter.size) / float(k)
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


def _result_file_name(
    name: str,
    N: int,
    D: int,
    Q: int,
    n_list: int,
    M: int,
    K: int,
    n_probe: int,
    top_k: int,
    scale_n: int,
    cluster_bound: int | None,
    layout: str | None,
) -> Path:
    filename = (
        f"acc_{name}"
        f"_N{N}_D{D}_Q{Q}"
        f"_nlist{n_list}_M{M}_K{K}"
        f"_nprobe{n_probe}_topk{top_k}_scale{scale_n}"
    )
    if cluster_bound is not None:
        filename += f"_cb{cluster_bound}"
    filename += layout_suffix(layout)
    filename += ".json"
    return RESULT_DIR / filename


def _load_cached(
    path: Path,
    *,
    report_ks: Tuple[int, ...],
    layout: str | None,
) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    config = payload.get("config")
    cached_report_ks: Tuple[int, ...] | None = None
    if isinstance(config, dict) and "report_ks" in config:
        try:
            cached_report_ks = tuple(sorted(int(k) for k in config["report_ks"]))
        except Exception:
            cached_report_ks = None

    if cached_report_ks is not None and cached_report_ks != tuple(sorted(report_ks)):
        return []

    cached_layout: str | None = None
    if isinstance(config, dict):
        try:
            cached_layout = normalize_layout(config.get("layout"))
        except Exception:
            return []
    if cached_layout != normalize_layout(layout):
        return []

    required_keys = set(_required_run_keys(report_ks))
    runs = payload.get("runs", [])
    out: List[Dict[str, float]] = []
    for run in runs:
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
    name: str,
    config: Dict[str, int | List[int] | None],
    runs: List[Dict[str, float]],
) -> None:
    payload = {
        "name": name,
        "metric_version": METRIC_VERSION,
        "config": config,
        "metrics": sorted(runs[0].keys()) if runs else list(_legacy_alias_keys()),
        "runs": runs,
    }
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

    summary: Dict[str, Dict[str, float]] = {}

    for group in ("standard", "zk"):
        group_summary: Dict[str, float] = {}
        for metric in ("pass", "recall"):
            for k in report_ks:
                run_key = _metric_key(group, metric, k)
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

                group_summary[f"mean_{metric}_at_{k}"] = mean
                group_summary[f"ci95_{metric}_at_{k}"] = ci95

        alias_mean_key = f"mean_pass_at_{top_k}"
        alias_ci_key = f"ci95_pass_at_{top_k}"
        if alias_mean_key in group_summary:
            group_summary["mean_pass_at_k"] = group_summary[alias_mean_key]
        if alias_ci_key in group_summary:
            group_summary["ci95_pass_at_k"] = group_summary[alias_ci_key]
            group_summary["ci95"] = group_summary[alias_ci_key]

        alias_mean_key = f"mean_recall_at_{top_k}"
        alias_ci_key = f"ci95_recall_at_{top_k}"
        if alias_mean_key in group_summary:
            group_summary["mean_recall_at_k"] = group_summary[alias_mean_key]
        if alias_ci_key in group_summary:
            group_summary["ci95_recall_at_k"] = group_summary[alias_ci_key]

        for metric_name, run_key in SUMMARY_EXTRA_KEYS.get(group, ()):
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

    shared_summary: Dict[str, float] = {}
    for metric_name, run_key in SUMMARY_EXTRA_KEYS.get("shared", ()):
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

        shared_summary[f"mean_{metric_name}"] = mean
        shared_summary[f"ci95_{metric_name}"] = ci95
    if shared_summary:
        summary["shared"] = shared_summary
    return summary


def _run_once(
    database_vecs: np.ndarray,
    query_vecs: np.ndarray,
    gt_vecs: np.ndarray | None,
    top_k: int,
    n_list: int,
    M: int,
    K: int,
    n_probe: int,
    scale_n: int,
    cluster_bound: int | None,
    report_ks: Tuple[int, ...],
    layout: str | None,
) -> Dict[str, float]:
    """
    进行一次训练 + 检索，对标准 IVF-PQ 与 ZK IVF-PQ
    从同一次 top_k 检索结果中同时计算多个 k 的 pass@k 与 recall@k。
    """
    base = np.asarray(database_vecs, dtype=np.float32)
    queries = np.asarray(query_vecs, dtype=np.float32)

    if base.ndim != 2 or queries.ndim != 2:
        raise ValueError("database_vecs and query_vecs must be 2D arrays")

    N, D = base.shape
    Q, Dq = queries.shape
    if D != Dq:
        raise ValueError(f"dimension mismatch: database D={D}, query D={Dq}")
    if top_k <= 0 or top_k > N:
        raise ValueError("top_k must be in [1, N]")

    # 计算 report_ks 中的最大值，查询时需要检索这么多结果
    max_report_k = int(max(report_ks)) if report_ks else top_k

    # 1. ground truth: 若提供 gt_vecs 则直接使用，否则回退到 brute-force 计算
    if gt_vecs is not None:
        gt_arr = np.asarray(gt_vecs)
        if gt_arr.ndim != 2:
            raise ValueError("gt_vecs must be a 2D array of shape (Q, K_gt)")
        Q_gt, K_gt = gt_arr.shape
        if Q_gt != Q:
            raise ValueError(f"gt_vecs Q mismatch: expected {Q}, got {Q_gt}")
        if max_report_k > K_gt:
            raise ValueError(
                f"max_report_k={max_report_k} exceeds available ground-truth size K_gt={K_gt}"
            )
        t0 = time.time()
        gt_topk = [gt_arr[i, :max_report_k] for i in range(Q)]
        bruteforce_time = time.time() - t0
        print(f"[acc_bench] bruteforce_time(precomputed_gt)={bruteforce_time:.3f}s")
    else:
        t0 = time.time()
        gt_topk = [brute_force_knn(base, queries[i], max_report_k) for i in range(Q)]
        bruteforce_time = time.time() - t0
        print(f"[acc_bench] bruteforce_time={bruteforce_time:.3f}s")

    # 为本次 run 生成不同的随机种子，使多次 run 之间有随机性
    rng = np.random.default_rng()
    std_seed = int(rng.integers(0, 2**31 - 1))
    zk_seed = int(rng.integers(0, 2**31 - 1))

    # 2. 非 ZK 版本：使用浮点 standard IVF-PQ
    t0 = time.time()
    std_labels, std_center, std_code_books, std_quant_vecs, std_id_groups = (
        ivf_pq_learn(
            base,
            n_list=n_list,
            M=M,
            K=K,
            random_state=std_seed,
            layout=layout,
        )
    )
    standard_train_time = time.time() - t0
    print(f"[acc_bench] standard_train_time={standard_train_time:.3f}s")

    std_metric_lists = {
        _metric_key("standard", metric, k): []
        for metric in ("pass", "recall")
        for k in report_ks
    }
    t0 = time.time()
    for i in tqdm(range(Q), "非zk版本"):
        pred = ivf_pq_query(
            queries[i],
            max_report_k,  # 查询时取 max_report_k 个结果
            std_labels,
            std_center,
            std_code_books,
            std_quant_vecs,
            std_id_groups,
            n_probe=n_probe,
            layout=layout,
        )
        _append_metric_lists(
            std_metric_lists,
            _query_metrics(
                scheme="standard",
                pred=pred,
                gt_topk=gt_topk[i],
                report_ks=report_ks,
            ),
        )
    standard_query_time = time.time() - t0
    print(f"[acc_bench] standard_query_time={standard_query_time:.3f}s")

    # 3. ZK 版本：首先 rescale，然后使用 zk 版本的 learn + query
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
            layout=layout,
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
            layout=layout,
        )

    # 计算 ZK 证明中每簇需要 padding 到的容量 n（power-of-two 容量）
    zk_n = _build_cluster_capacity(zk_id_groups, n_probe)
    zk_train_time = time.time() - t0
    print(f"[acc_bench] zk_train_time={zk_train_time:.3f}s")

    zk_metric_lists = {
        _metric_key("zk", metric, k): []
        for metric in ("pass", "recall")
        for k in report_ks
    }
    t0 = time.time()
    for i in tqdm(range(Q), "zk版本"):
        scaled_query = rescale_query(queries[i], scale_n, v_min, v_max)
        pred_zk, _ = zk_ivf_pq_query(
            scaled_query,
            zk_center,
            zk_code_books,
            zk_quant_vecs,
            zk_id_groups,
            top_k=max_report_k,  # 查询时取 max_report_k 个结果
            n_probe=n_probe,
            proof=False,
            layout=layout,
        )
        _append_metric_lists(
            zk_metric_lists,
            _query_metrics(
                scheme="zk",
                pred=pred_zk,
                gt_topk=gt_topk[i],
                report_ks=report_ks,
            ),
        )
    zk_query_time = time.time() - t0
    print(f"[acc_bench] zk_query_time={zk_query_time:.3f}s")

    result = {
        "zk_n": float(zk_n),
        "bruteforce_time": float(bruteforce_time),
        "standard_train_time": float(standard_train_time),
        "standard_query_time": float(standard_query_time),
        "zk_train_time": float(zk_train_time),
        "zk_query_time": float(zk_query_time),
    }
    result.update(_mean_metric_lists(std_metric_lists))
    result.update(_mean_metric_lists(zk_metric_lists))
    _add_legacy_aliases(result, top_k)
    if zk_changed_count is not None:
        result["zk_changed_count"] = float(zk_changed_count)
    return result


def run_accuracy_bench(
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
    report_ks: Iterable[int] | None = None,
    layout: str | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    比较非 ZK 与 ZK IVF-PQ 的检索质量，并在 data/acc_bench 下缓存结果。

    参数:
        database_vecs: (N, D) 向量库
        query_vecs: (Q, D) 查询向量
        gt_vecs: (Q, K_gt) 预先计算好的 ground truth 邻居索引（例如 SIFT 的前 100 个真近邻）
        top_k: 单次检索返回的最大 K（需满足 top_k <= K_gt）
        name: 用于缓存文件名的前缀，方便区分实验
        n_list, M, K, n_probe: IVF-PQ 超参数
        scale_n: ZK 版本 rescale 时的整数上界
        cluster_bound: 若不为 None, 则 ZK 版本训练时对 coarse 簇大小施加上界
        num_runs: 训练 / 评估重复次数，用于估计 95% CI
        force_recompute: 若为 True，则忽略已缓存结果，重新计算所有 run
        report_ks: 需要汇报的多个 k。若为 None，则默认汇报 1/10/50/100，
            并自动补上 top_k 本身（若不在其中）。
        layout: 向量维度布局；None 表示保持原始顺序，"mod8" 表示论文中的模 8 重排。
    返回:
        {
            "standard": {
                "mean_pass_at_1": ...,
                "mean_recall_at_1": ...,
                ...,
            },
            "zk": {
                "mean_pass_at_1": ...,
                "mean_recall_at_1": ...,
                ...,
            },
        }
    """
    base = np.asarray(database_vecs)
    queries = np.asarray(query_vecs)
    if base.ndim != 2 or queries.ndim != 2:
        raise ValueError("database_vecs and query_vecs must be 2D arrays")

    N, D = base.shape
    Q, Dq = queries.shape
    if D != Dq:
        raise ValueError(f"dimension mismatch: database D={D}, query D={Dq}")

    gt_arr = np.asarray(gt_vecs)
    if gt_arr.ndim != 2:
        raise ValueError("gt_vecs must be a 2D array of shape (Q, K_gt)")
    Q_gt, K_gt = gt_arr.shape
    if Q_gt != Q:
        raise ValueError(f"gt_vecs Q mismatch: expected {Q}, got {Q_gt}")
    if top_k > K_gt:
        raise ValueError(
            f"top_k={top_k} exceeds available ground-truth size K_gt={K_gt}"
        )

    resolved_report_ks = _normalize_report_ks(top_k, report_ks)
    resolved_layout = normalize_layout(layout)
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
        scale_n=scale_n,
        cluster_bound=cluster_bound,
        layout=resolved_layout,
    )

    if force_recompute:
        runs: List[Dict[str, float]] = []
    else:
        runs = _load_cached(
            path,
            report_ks=resolved_report_ks,
            layout=resolved_layout,
        )
        if path.exists() and not runs:
            print(
                "[acc_bench] cache ignored "
                f"(missing required metrics or layout/report_ks mismatch): {path}"
            )

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
                    resolved_report_ks,
                    resolved_layout,
                )
            )
        config = {
            "N": N,
            "D": D,
            "Q": Q,
            "n_list": n_list,
            "M": M,
            "K": K,
            "n_probe": n_probe,
            "top_k": top_k,
            "report_ks": [int(k) for k in resolved_report_ks],
            "scale_n": scale_n,
            "cluster_bound": cluster_bound,
            "layout": resolved_layout,
        }
        _save_cached(path, name, config, runs)

    summary = _compute_summary(runs, top_k=top_k, report_ks=resolved_report_ks)
    return summary
