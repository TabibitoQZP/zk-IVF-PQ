import numpy as np
from typing import Tuple

MAX_SCALE = 65536


def brute_force_knn(
    base_vecs: np.ndarray, query_vec: np.ndarray, top_k: int
) -> np.ndarray:
    """
    使用暴力枚举的方式，以 L2 距离计算 KNN 结果，作为 ground truth。
    """
    base_vecs = np.asarray(base_vecs)
    query_vec = np.asarray(query_vec, dtype=base_vecs.dtype)

    if base_vecs.ndim != 2:
        raise ValueError("base_vecs must be a 2D array of shape (N, D)")
    if query_vec.ndim != 1:
        raise ValueError("query_vec must be a 1D array of shape (D,)")

    N, D = base_vecs.shape
    if query_vec.shape[0] != D:
        raise ValueError(f"dimension mismatch: base_vecs D={D}, query_vec D={query_vec.shape[0]}")

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k > N:
        top_k = N

    diff = base_vecs - query_vec
    dist2 = np.sum(diff * diff, axis=1)

    # 使用 argpartition 先选出前 top_k 小的元素，再在这 top_k 内做完整排序，
    # 将复杂度从 O(N log N) 降为 O(N) + O(top_k log top_k)，在 N 很大且 top_k 很小时明显加速。
    if top_k == N:
        topk_idx = np.argsort(dist2)
    else:
        partial = np.argpartition(dist2, top_k - 1)[:top_k]
        order = np.argsort(dist2[partial])
        topk_idx = partial[order]

    return topk_idx.astype(np.int64)


def rescale_database(
    vecs: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, float, float]:
    """
    将数据库向量整体线性缩放到整数区间 [0, n)。

    返回:
        scaled_vecs: 与 vecs 同形状的 int64 数组，元素范围在 [0, n)
        v_min: 缩放前全局最小值 (float)
        v_max: 缩放前全局最大值 (float)
    """
    if n <= 1:
        raise ValueError("n must be greater than 1 for rescaling.")

    arr = np.asarray(vecs, dtype=np.float64)
    if arr.size == 0:
        return arr.astype(np.int64), 0.0, 0.0

    v_min = float(arr.min())
    v_max = float(arr.max())

    if v_max == v_min:
        # 所有值相同，全部映射为 0
        scaled = np.zeros_like(arr, dtype=np.int64)
        return scaled, v_min, v_max

    # 线性映射到 [0, n-1]，然后取整并截断，保证在 [0, n-1] 内
    scale = (n - 1) / (v_max - v_min)
    scaled_float = (arr - v_min) * scale
    scaled_int = np.rint(scaled_float).astype(np.int64)
    scaled_int = np.clip(scaled_int, 0, n - 1)

    return scaled_int, v_min, v_max


def rescale_query(
    query: np.ndarray,
    n: int,
    v_min: float,
    v_max: float,
) -> np.ndarray:
    """
    使用数据库同样的缩放参数，将查询向量缩放到整数区间 [0, n)。
    """
    if n <= 1:
        raise ValueError("n must be greater than 1 for rescaling.")

    q = np.asarray(query, dtype=np.float64)
    if v_max == v_min:
        return np.zeros_like(q, dtype=np.int64)

    scale = (n - 1) / (v_max - v_min)
    scaled_float = (q - v_min) * scale
    scaled_int = np.rint(scaled_float).astype(np.int64)
    scaled_int = np.clip(scaled_int, 0, n - 1)

    return scaled_int
