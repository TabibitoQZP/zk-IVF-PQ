import numpy as np
from ivf_pq.util.kmeans import kmeans_with_ids

from zk_IVF_PQ.zk_IVF_PQ import py_ivf_pq_verify_proof


def ivf_pq_learn(
    vecs: np.ndarray,
    n_list=64,
    n_iter=64,
    M=8,
    K=256,
):
    N, D = vecs.shape

    d = D // M
    assert d * M == D, "M 应当被D整除"

    center, id_groups, labels = kmeans_with_ids(vecs, n_list, n_iter)

    centers = center[labels]

    res_vecs = vecs - centers

    code_books = []
    quant_vecs = []
    for i in range(0, D, d):
        res_slide = res_vecs[:, i : i + d]
        slide_center, slide_id_groups, slide_labels = kmeans_with_ids(
            res_slide, K, n_iter
        )
        code_books.append(slide_center)
        quant_vecs.append(slide_labels)

    code_books = np.array(code_books)
    quant_vecs = np.ascontiguousarray(np.array(quant_vecs).T)

    # (N,), (n_list,D), (M,K,D/M), (N,M)
    return labels, center, code_books, quant_vecs, id_groups


def upperbound(id_groups, n_probe=8):
    size = [v.shape[0] for v in id_groups.values()]
    size.sort()
    return sum(size[-n_probe:])


def zk_ivf_pq_query(
    query,  # to use
    center,  # ivf_centers
    code_books,  # codebooks
    quant_vecs,
    id_groups,
    n_probe=8,
):
    max_sz = upperbound(id_groups, n_probe=n_probe)
    M, K, _ = code_books.shape
    diff = query - center  # (n, d)，广播减法
    dist2 = (diff * diff).sum(axis=1, dtype=np.int64)
    sorted_idx = np.argsort(dist2)
    sorted_center = center[sorted_idx, :]
    sorted_dist2 = dist2[sorted_idx]
    sorted_idx_dis = np.stack([sorted_idx, sorted_dist2], axis=1)  # to use

    # NOTE: 注意res有正有负, 要开始小心处理
    res = query - sorted_center[:n_probe]
    filtered_centers = sorted_center[:n_probe]  # to use

    filtered_ves = []
    filtered_dis = []
    filtered_idx = []
    sec_idx = []
    probe_count = []
    for si, idx in enumerate(sorted_idx[:n_probe]):
        vec_idx = id_groups[idx]
        curr_res = res[si]
        probe_count.append(len(vec_idx))
        for vi in vec_idx:
            sec_idx.append(si)
            filtered_idx.append(vi)
            q_vec = quant_vecs[vi]
            filtered_ves.append(q_vec)
            orig_vec = code_books[np.arange(M), q_vec, :].reshape(-1)
            sub_val = curr_res - orig_vec
            curr_dist2 = (sub_val * sub_val).sum(dtype=np.int64)
            filtered_dis.append(curr_dist2)

    filtered_dis = np.array(filtered_dis)
    filtered_sorted_idx = np.argsort(filtered_dis)
    filtered_dis = np.array(filtered_dis)[filtered_sorted_idx]
    filtered_idx = np.array(filtered_idx)[filtered_sorted_idx]
    filtered_ves = np.array(filtered_ves)[filtered_sorted_idx]  # filtered_vecs
    sec_idx = np.array(sec_idx)[filtered_sorted_idx]
    probe_count = np.array(probe_count)  # to use

    # TODO: 计算one-hot
    valid_vec_count = len(filtered_idx)
    orig_one_hot = np.zeros([valid_vec_count, n_probe], dtype=np.int64)
    orig_one_hot[np.arange(valid_vec_count), sec_idx] = 1
    vecs_cluster_hot = np.zeros([max_sz, n_probe], dtype=np.int64)
    vecs_cluster_hot[:valid_vec_count, :] = orig_one_hot
    # filtered_vecs也要扩展
    extend_filtered_vecs = np.zeros([max_sz, M], dtype=np.int64)
    extend_filtered_vecs[:valid_vec_count, :] = filtered_ves

    # TODO: 证明
    result = py_ivf_pq_verify_proof(
        center,
        query,
        sorted_idx_dis,
        filtered_centers,
        probe_count,
        extend_filtered_vecs,
        vecs_cluster_hot,
        code_books,
    )
    print(result)

    return filtered_idx


if __name__ == "__main__":
    # NOTE: 现在开始要很注意类型, 虽然都是大于0, 但在传入前一直以int64保存
    vecs = np.random.randint(0, 16383, size=[10000, 128], dtype=np.int64)
    query = np.random.randint(0, 16383, size=128, dtype=np.int64)
    labels, center, code_books, quant_vecs, id_groups = ivf_pq_learn(vecs)

    idxes = zk_ivf_pq_query(query, center, code_books, quant_vecs, id_groups)
    print(idxes[:16])
