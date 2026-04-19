import numpy as np

from ivf_pq.layout import apply_layout
from ivf_pq.util.kmeans import faiss_kmeans_with_ids


def ivf_pq_learn(
    vecs: np.ndarray,
    n_list: int = 64,
    n_iter: int = 25,
    M: int = 8,
    K: int = 256,
    random_state: int | None = 1234,
    layout: str | None = None,
):
    """
    使用浮点数实现的标准 IVF-PQ 训练流程（无零知识相关约束）。

    返回:
        labels: (N,) 每个向量所属的 coarse 簇编号
        center: (n_list, D) coarse 聚类中心（float32）
        code_books: (M, K, d) 子空间码本（float32）
        quant_vecs: (N, M) 每个向量在各子空间的码字索引（int64）
        id_groups: {cluster_id: np.ndarray[ids]} 每个 coarse 簇中的向量 id
    """
    vecs = apply_layout(np.asarray(vecs, dtype=np.float32), layout)
    if vecs.ndim != 2:
        raise ValueError("vecs must be 2D array of shape (N, D)")

    N, D = vecs.shape
    if M <= 0:
        raise ValueError("M must be positive")
    if K <= 0:
        raise ValueError("K must be positive")
    if n_list <= 0:
        raise ValueError("n_list must be positive")

    d = D // M
    if d * M != D:
        raise ValueError("M must divide D exactly")

    # 1. coarse 聚类（IVF）
    center, id_groups, labels = faiss_kmeans_with_ids(
        vecs,
        n_list,
        niter=n_iter,
        random_state=random_state,
    )
    center = center.astype(np.float32, copy=False)
    labels = labels.astype(np.int64, copy=False)

    # 每个向量对应的 coarse center
    centers_for_vecs = center[labels]  # (N, D)

    # 2. residual 计算
    res_vecs = vecs - centers_for_vecs

    # 3. 子空间 KMeans（PQ）
    code_books = []
    quant_vecs = []
    for offset in range(0, D, d):
        res_slice = res_vecs[:, offset : offset + d]
        slice_center, _, slice_labels = faiss_kmeans_with_ids(
            res_slice,
            K,
            niter=n_iter,
            random_state=random_state,
        )
        code_books.append(slice_center.astype(np.float32, copy=False))
        quant_vecs.append(slice_labels.astype(np.int64, copy=False))

    code_books_arr = np.stack(code_books, axis=0)  # (M, K, d)
    quant_vecs_arr = np.stack(quant_vecs, axis=1)  # (N, M)

    return labels, center, code_books_arr, quant_vecs_arr, id_groups


def ivf_pq_query(
    query: np.ndarray,
    top_k: int,
    labels: np.ndarray,
    center: np.ndarray,
    code_books: np.ndarray,
    quant_vecs: np.ndarray,
    id_groups: dict,
    n_probe: int = 8,
    layout: str | None = None,
) -> np.ndarray:
    """
    标准 IVF-PQ 检索（浮点数版本），返回近似 top_k 邻居索引。
    """
    query = apply_layout(np.asarray(query, dtype=np.float32), layout)
    center = np.asarray(center, dtype=np.float32)
    code_books = np.asarray(code_books, dtype=np.float32)
    quant_vecs = np.asarray(quant_vecs, dtype=np.int64)

    if query.ndim != 1:
        raise ValueError("query must be 1D array of shape (D,)")

    n_list, D = center.shape
    M, K, d = code_books.shape
    if M * d != D:
        raise ValueError("code_books shape is inconsistent with center dimension")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    # 1. 选择要探测的 coarse 簇
    diff = center - query  # (n_list, D)
    dist2 = np.einsum("ij,ij->i", diff, diff)
    cluster_order = np.argsort(dist2, kind="stable")

    # 2. 在前 n_probe 个簇中，根据 PQ 残差距离进行近似 KNN
    all_ids = []
    all_dis2 = []

    m_indices = np.arange(M)[None, :]  # 用于广播索引子空间

    for ci in cluster_order[: min(n_probe, n_list)]:
        ci_int = int(ci)
        # 该簇中的向量 id
        ids = id_groups.get(ci_int)
        if ids is None or ids.size == 0:
            continue

        curr_center = center[ci_int]  # (D,)
        res_query = query - curr_center  # (D,)

        curr_vec_codes = quant_vecs[ids]  # (N_c, M)

        # 通过 code_books 重建子空间残差向量
        # code_books: (M, K, d)
        # curr_vec_codes: (N_c, M)
        # 索引后得到 (N_c, M, d)，再 reshape 为 (N_c, D)
        curr_code_vecs = code_books[m_indices, curr_vec_codes, :]  # (N_c, M, d)
        curr_code_vecs = curr_code_vecs.reshape(curr_vec_codes.shape[0], -1)

        curr_diff = curr_code_vecs - res_query  # (N_c, D)
        curr_dis2 = np.einsum("ij,ij->i", curr_diff, curr_diff)

        all_ids.append(ids)
        all_dis2.append(curr_dis2)

    if not all_ids:
        return np.empty((0,), dtype=np.int64)

    all_ids_arr = np.concatenate(all_ids)
    all_dis2_arr = np.concatenate(all_dis2)

    order = np.argsort(all_dis2_arr, kind="stable")
    top_k = min(top_k, order.size)
    return all_ids_arr[order[:top_k]].astype(np.int64)
