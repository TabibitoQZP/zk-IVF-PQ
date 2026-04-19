import numpy as np

from ivf_pq.layout import apply_layout
from ivf_pq.zk import ivf_pq_learn, upperbound
from zk_IVF_PQ.zk_IVF_PQ import py_set_based_with_merkle, single_hash


def _build_cluster_capacity(id_groups, n_probe: int) -> int:
    """Return per-cluster capacity (power of two >= max cluster size)."""
    sizes = [group.shape[0] for group in id_groups.values()]
    if not sizes:
        return 1
    max_size = max(sizes)
    capacity = 1
    while capacity < max_size:
        capacity *= 2
    return capacity


def _compute_cluster_root(
    cluster_index: int,
    vpqs: np.ndarray,  # (capacity, M)
    valids: np.ndarray,  # (capacity,)
    items: np.ndarray,  # (capacity,)
) -> np.uint64:
    """
    计算单个 IVF 簇在 (cluster_idx, j, valid, item, vpqs[j]) 语义下的 Merkle 根。
    逻辑应与 Rust 端 merkle_cluster_i64 / merkle_cluster_gadget 保持一致。
    """
    capacity, _ = vpqs.shape
    hash_list: list[int] = []

    for j in range(capacity):
        left = np.array(
            [cluster_index, j, int(valids[j]), int(items[j])],
            dtype=np.int64,
        )
        leaf = np.concatenate([left, vpqs[j].astype(np.int64)])
        hash_list.append(single_hash(leaf))

    while len(hash_list) > 1:
        hash_list = [
            single_hash([hash_list[2 * i], hash_list[2 * i + 1]])
            for i in range(len(hash_list) // 2)
        ]

    return np.uint64(hash_list[0])


def zk_ivf_pq_query(
    query,  # (D,)
    center,  # (n_list, D)
    code_books,  # (M, K, d)
    quant_vecs,  # (N, M)
    id_groups,
    top_k: int = 10,
    n_probe: int = 8,
    proof: bool = False,
    layout: str | None = None,
):
    """
    IVF-PQ 查询。

    - 当 proof=False 时，仅执行近似检索（不构造 Merkle 结构、不生成证明）。
    - 当 proof=True 时，构造带 Merkle 承诺的 set-based 证明系统所需的结构，并返回证明指标。
    """
    query = apply_layout(np.asarray(query, dtype=np.int64), layout)
    center = np.asarray(center, dtype=np.int64)
    code_books = np.rint(code_books).astype(np.int64)
    quant_vecs = np.asarray(quant_vecs, dtype=np.int64)

    n_list, dim = center.shape
    m, k, d = code_books.shape
    assert m * d == dim, "code_books 形状与向量维度不匹配"

    # 1. 计算 IVF 簇排序与 cluster_idx_dis
    diff = center - query
    dist2 = (diff * diff).sum(axis=1, dtype=np.int64)
    sorted_idx = np.argsort(dist2, kind="stable")
    sorted_dist2 = dist2[sorted_idx]
    cluster_idx_dis = np.stack([sorted_idx, sorted_dist2], axis=1).astype(np.int64)
    cluster_idxes = sorted_idx[:n_probe]

    # Fast path: 只做近似检索，不构造 Merkle 结构
    if not proof:
        m_indices = np.arange(m)[None, :]  # (1, M)，用于广播索引子空间
        all_ids = []
        all_dis2 = []

        for cluster_index in cluster_idxes:
            ci_int = int(cluster_index)
            vector_ids = id_groups.get(ci_int)
            if vector_ids is None or vector_ids.size == 0:
                continue

            delta_query = query - center[ci_int]  # (D,)
            curr_codes = quant_vecs[vector_ids]  # (N_c, M)

            # 通过 code_books 重建子空间残差向量
            # code_books: (M, K, d)
            # curr_codes: (N_c, M)
            curr_code_vecs = code_books[m_indices, curr_codes, :]  # (N_c, M, d)
            curr_code_vecs = curr_code_vecs.reshape(curr_codes.shape[0], -1)  # (N_c, D)

            curr_diff = delta_query - curr_code_vecs  # (N_c, D)
            curr_dis2 = np.einsum("ij,ij->i", curr_diff, curr_diff)

            all_ids.append(vector_ids)
            all_dis2.append(curr_dis2)

        if not all_ids:
            return np.empty((0,), dtype=np.int64), None

        all_ids_arr = np.concatenate(all_ids)
        all_dis2_arr = np.concatenate(all_dis2)

        order = np.argsort(all_dis2_arr, kind="stable")
        top_k = min(top_k, order.size)
        top_k_items = all_ids_arr[order[:top_k]].astype(np.int64)
        return top_k_items, None

    # 2. 构造 vpqss / valids / itemss（用于 Merkle 承诺与证明）
    capacity = _build_cluster_capacity(id_groups, n_probe)
    vpqss_list = []
    valids_list = []
    itemss_list = []
    ivf_roots = np.zeros((n_list,), dtype=np.uint64)
    visited = np.zeros((n_list,), dtype=bool)

    # 2.1 为将参与检索的 n_probe 个簇构造 vpqss / valids / itemss，并计算对应 Merkle 根
    for cluster_index in cluster_idxes:
        cluster_index_int = int(cluster_index)
        vector_ids = id_groups[cluster_index_int]

        vpqs = np.zeros((capacity, m), dtype=np.int64)
        valids = np.zeros((capacity,), dtype=np.int64)
        items = np.zeros((capacity,), dtype=np.int64)

        for local_pos, vec_id in enumerate(vector_ids):
            if local_pos >= capacity:
                break
            vec_id_int = int(vec_id)
            items[local_pos] = vec_id_int
            valids[local_pos] = 1
            vpqs[local_pos, :] = quant_vecs[vec_id_int]

        vpqss_list.append(vpqs)
        valids_list.append(valids)
        itemss_list.append(items)

        ivf_roots[cluster_index_int] = _compute_cluster_root(
            cluster_index_int, vpqs, valids, items
        )
        visited[cluster_index_int] = True

    # 2.2 为其余簇计算 Merkle 根，使得 ivf_roots 对整棵 IVF 树形成一致的承诺
    for cluster_index_int in range(n_list):
        if visited[cluster_index_int]:
            continue

        vector_ids = id_groups[cluster_index_int]
        vpqs = np.zeros((capacity, m), dtype=np.int64)
        valids = np.zeros((capacity,), dtype=np.int64)
        items = np.zeros((capacity,), dtype=np.int64)

        for local_pos, vec_id in enumerate(vector_ids):
            if local_pos >= capacity:
                break
            vec_id_int = int(vec_id)
            items[local_pos] = vec_id_int
            valids[local_pos] = 1
            vpqs[local_pos, :] = quant_vecs[vec_id_int]

        ivf_roots[cluster_index_int] = _compute_cluster_root(
            cluster_index_int, vpqs, valids, items
        )

    vpqss = np.stack(vpqss_list, axis=0)  # (n_probe, capacity, M)
    valids = np.stack(valids_list, axis=0)  # (n_probe, capacity)
    itemss = np.stack(itemss_list, axis=0)  # (n_probe, capacity)

    # 3. 使用与证明系统一致的方式计算近似距离并取 top_k
    max_dis = (1 << 62) - 1
    all_item_dis = []
    for probe_pos, cluster_index in enumerate(cluster_idxes):
        cluster_index_int = int(cluster_index)
        delta_query = query - center[cluster_index_int]
        for local_pos in range(capacity):
            code_indices = vpqss[probe_pos, local_pos]
            selected = code_books[np.arange(m), code_indices]
            code_vec = selected.reshape(m * d)
            curr_diff = delta_query - code_vec
            curr_dis = int(np.dot(curr_diff, curr_diff))
            if valids[probe_pos, local_pos] == 0:
                curr_dis = max_dis
            all_item_dis.append((int(itemss[probe_pos, local_pos]), curr_dis))

    all_item_dis = np.array(all_item_dis, dtype=np.int64)
    sort_idx = np.argsort(all_item_dis[:, 1], kind="stable")
    all_item_dis = all_item_dis[sort_idx]
    top_k_items = all_item_dis[:top_k, 0]

    # 4. 调用带 Merkle 承诺的证明系统
    result = py_set_based_with_merkle(
        query,
        center,
        vpqss,
        valids,
        itemss,
        code_books,
        ivf_roots,
        int(top_k),
        cluster_idx_dis,
        [],  # ordered_vpqss_item_dis 由 Rust 内部重新计算
    )
    print("Merkle ZK proof metrics:", result)

    return top_k_items, result


if __name__ == "__main__":
    # 简单自测：随机数据跑通一次流程
    n_list = 32
    N = 10000
    vecs = np.random.randint(0, 16383, size=[N, 128], dtype=np.int64)
    query = np.random.randint(0, 16383, size=128, dtype=np.int64)
    labels, center, code_books, quant_vecs, id_groups = ivf_pq_learn(
        vecs, n_list=n_list
    )
    max_val = 0
    for k, v in id_groups.items():
        max_val = max(max_val, len(v))
    print(max_val, N / n_list)

    indices, _ = zk_ivf_pq_query(
        query, center, code_books, quant_vecs, id_groups, proof=True
    )
    print(indices[:16])
