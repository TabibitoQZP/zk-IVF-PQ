import numpy as np

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
    proof=False,
):
    """
    IVF-PQ 查询，并使用带 Merkle 承诺的 set-based 证明系统进行验证。
    返回近似检索到的 top_k 向量索引。
    """
    query = np.asarray(query, dtype=np.int64)
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

    # 2. 构造 vpqss / valids / itemss
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
    result = None
    if proof:
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
    vecs = np.random.randint(0, 16383, size=[10000, 128], dtype=np.int64)
    query = np.random.randint(0, 16383, size=128, dtype=np.int64)
    labels, center, code_books, quant_vecs, id_groups = ivf_pq_learn(vecs)

    indices, _ = zk_ivf_pq_query(query, center, code_books, quant_vecs, id_groups)
    print(indices[:16])
