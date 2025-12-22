import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

import numpy as np

from ivf_pq import MAX_SCALE, brute_force_knn, rescale_database, rescale_query
from ivf_pq.merkle_zk import (
    ivf_pq_learn as zk_ivf_pq_learn,
    zk_ivf_pq_query,
    _build_cluster_capacity,
)
from ivf_pq.standard import ivf_pq_learn, ivf_pq_query


RESULT_DIR = Path("data") / "acc_bench"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

METRIC_KEYS: Tuple[str, str] = ("standard_pass_at_k", "zk_pass_at_k")


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
) -> Path:
    filename = (
        f"acc_{name}"
        f"_N{N}_D{D}_Q{Q}"
        f"_nlist{n_list}_M{M}_K{K}"
        f"_nprobe{n_probe}_topk{top_k}_scale{scale_n}"
    )
    if cluster_bound is not None:
        filename += f"_cb{cluster_bound}"
    filename += ".json"
    return RESULT_DIR / filename


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
                "standard_pass_at_k": float(run["standard_pass_at_k"]),
                "zk_pass_at_k": float(run["zk_pass_at_k"]),
            }
            # 可选保存的附加信息（每簇 padding 长度、各阶段时间），若存在则保留
            optional_keys = [
                "zk_n",
                "bruteforce_time",
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


def _compute_summary(
    runs: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    if not runs:
        raise ValueError("No runs provided for summary")

    values = np.array(
        [[run["standard_pass_at_k"], run["zk_pass_at_k"]] for run in runs],
        dtype=float,
    )
    means = values.mean(axis=0)
    if len(runs) > 1:
        std = values.std(axis=0, ddof=1)
        ci95 = 1.96 * std / math.sqrt(len(runs))
    else:
        ci95 = np.zeros_like(means)

    return {
        "standard": {
            "mean_pass_at_k": float(means[0]),
            "ci95": float(ci95[0]),
        },
        "zk": {
            "mean_pass_at_k": float(means[1]),
            "ci95": float(ci95[1]),
        },
    }


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
) -> Dict[str, float]:
    """
    进行一次训练 + 检索，对标准 IVF-PQ 与 ZK IVF-PQ
    分别计算相对于暴力 L2 KNN 的 pass@k。
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

    # 1. ground truth: 若提供 gt_vecs 则直接使用，否则回退到 brute-force 计算
    if gt_vecs is not None:
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
        t0 = time.time()
        gt_topk = [gt_arr[i, :top_k] for i in range(Q)]
        bruteforce_time = time.time() - t0
        print(f"[acc_bench] bruteforce_time(precomputed_gt)={bruteforce_time:.3f}s")
    else:
        t0 = time.time()
        gt_topk = [brute_force_knn(base, queries[i], top_k) for i in range(Q)]
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
        )
    )
    standard_train_time = time.time() - t0
    print(f"[acc_bench] standard_train_time={standard_train_time:.3f}s")

    std_pass_list: List[float] = []
    std_recall_list: List[float] = []
    t0 = time.time()
    for i in tqdm(range(Q), "非zk版本"):
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
        inter = np.intersect1d(pred, gt_topk[i])
        std_pass_list.append(float(inter.size) / float(top_k))
        best_gt = int(gt_topk[i][0])
        std_recall_list.append(1.0 if best_gt in pred else 0.0)
    standard_query_time = time.time() - t0
    print(f"[acc_bench] standard_query_time={standard_query_time:.3f}s")

    # 3. ZK 版本：首先 rescale，然后使用 zk 版本的 learn + query
    t0 = time.time()
    scaled_base, v_min, v_max = rescale_database(base, scale_n)
    if cluster_bound is not None:
        (
            zk_labels,
            zk_center,
            zk_code_books,
            zk_quant_vecs,
            zk_id_groups,
            _changed_count,
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
    print(f"[acc_bench] zk_train_time={zk_train_time:.3f}s")

    zk_pass_list: List[float] = []
    zk_recall_list: List[float] = []
    t0 = time.time()
    for i in tqdm(range(Q), "zk版本"):
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
        inter = np.intersect1d(pred_zk, gt_topk[i])
        zk_pass_list.append(float(inter.size) / float(top_k))
        best_gt = int(gt_topk[i][0])
        zk_recall_list.append(1.0 if best_gt in pred_zk else 0.0)
    zk_query_time = time.time() - t0
    print(f"[acc_bench] zk_query_time={zk_query_time:.3f}s")

    result = {
        "standard_pass_at_k": float(np.mean(std_pass_list)),
        "zk_pass_at_k": float(np.mean(zk_pass_list)),
        "standard_recall_at_k": float(np.mean(std_recall_list)),
        "zk_recall_at_k": float(np.mean(zk_recall_list)),
        "zk_n": float(zk_n),
        "bruteforce_time": float(bruteforce_time),
        "standard_train_time": float(standard_train_time),
        "standard_query_time": float(standard_query_time),
        "zk_train_time": float(zk_train_time),
        "zk_query_time": float(zk_query_time),
    }
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
) -> Dict[str, Dict[str, float]]:
    """
    比较非 ZK 与 ZK IVF-PQ 的准确率（pass@k），并在 data/acc_bench 下缓存结果。

    参数:
        database_vecs: (N, D) 向量库
        query_vecs: (Q, D) 查询向量
        gt_vecs: (Q, K_gt) 预先计算好的 ground truth 邻居索引（例如 SIFT 的前 100 个真近邻）
        top_k: 评估的 K（需满足 top_k <= K_gt）
        name: 用于缓存文件名的前缀，方便区分实验
        n_list, M, K, n_probe: IVF-PQ 超参数
        scale_n: ZK 版本 rescale 时的整数上界
        cluster_bound: 若不为 None, 则 ZK 版本训练时对 coarse 簇大小施加上界
        num_runs: 训练 / 评估重复次数，用于估计 95% CI
        force_recompute: 若为 True，则忽略已缓存结果，重新计算所有 run
    返回:
        {
            "standard": {"mean_pass_at_k": ..., "ci95": ...},
            "zk": {"mean_pass_at_k": ..., "ci95": ...},
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
            "n_list": n_list,
            "M": M,
            "K": K,
            "n_probe": n_probe,
            "top_k": top_k,
            "scale_n": scale_n,
            "cluster_bound": cluster_bound,
        }
        _save_cached(path, name, config, runs)

    summary = _compute_summary(runs)
    return summary
