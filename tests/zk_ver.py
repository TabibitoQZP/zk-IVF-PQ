import time
import numpy as np
from zk_IVF_PQ.zk_IVF_PQ import single_hash, verify_ids_sorted_by_distance
from vec_data_load.sift import SIFT

from ivf_pq.pipeline import ivf_pq_learn


def merkle_build(data: list):
    sz = 1
    while len(data) > sz:
        sz *= 2

    b_tree = []
    filted_data = data + [[0] for _ in range(sz - len(data))]

    b_tree = [single_hash(item) for item in filted_data]
    sz //= 2

    while sz > 0:
        curr_level = []
        for i in range(sz):
            curr_level.append(single_hash([b_tree[2 * i], b_tree[2 * i + 1]]))
        b_tree = curr_level + b_tree
        sz //= 2

    return b_tree, filted_data


def zk_ivf_pq_query(
    query: np.ndarray,
    top_k: int,
    labels: np.ndarray,
    quant_center: np.ndarray,
    quant_code_books: np.ndarray,
    quant_vecs: np.ndarray,
    id_groups: dict,
    n_probe: int = 8,
):
    # step 1
    diff = quant_center - query
    dis_2 = np.einsum("ij,ij->i", diff, diff)
    cluster_idx = np.argsort(dis_2, kind="stable")

    print("Prove start.")
    s_time = time.time()
    data = verify_ids_sorted_by_distance(quant_center, query, cluster_idx)
    print(data, time.time() - s_time)

    # step 2
    all_cluster_idxes = []
    all_vec_idxes = []
    all_dis_2 = []
    for i in cluster_idx[:n_probe]:
        curr_center = quant_center[i]
        res_query = query - curr_center
        ids = id_groups[i]

        curr_N = ids.shape[0]

        curr_vecs = quant_vecs[ids]
        curr_q_vecs = quant_code_books[
            np.arange(curr_vecs.shape[1])[None, :], curr_vecs, :
        ].reshape(curr_vecs.shape[0], -1)

        curr_diff = curr_q_vecs - res_query
        curr_dis_2 = np.einsum("ij,ij->i", curr_diff, curr_diff)

        all_cluster_idxes.append(np.full((curr_N,), i, dtype=np.int32))
        all_vec_idxes.append(ids)
        all_dis_2.append(curr_dis_2)

    all_cluster_idxes = np.concatenate(all_cluster_idxes)
    all_vec_idxes = np.concatenate(all_vec_idxes)
    all_dis_2 = np.concatenate(all_dis_2)
    all_sorted_idx = np.argsort(all_dis_2, kind="stable")

    all_cluster_idxes = all_cluster_idxes[all_sorted_idx]
    all_vec_idxes = all_vec_idxes[all_sorted_idx]
    all_dis_2 = all_dis_2[all_sorted_idx]

    return all_vec_idxes[:top_k]


if __name__ == "__main__":
    top_k = 10
    data_root = "data/siftsmall/"
    sift = SIFT(data_root)

    vecs = sift.base_vecs  # (N,D)
    query_vecs = sift.query_vecs  # (100,D)
    gt_vecs = sift.gt_vecs  # (100,100)

    print("IVF-PQ learning start.")
    labels, quant_center, quant_code_books, quant_vecs, id_groups = ivf_pq_learn(
        vecs,
        # n_list=1024,
        # n_iter=16,
    )

    print("IVF-PQ learning finished.")

    query = query_vecs[0]

    print(quant_center.shape)
    b_tree = merkle_build(quant_center.tolist())
    query = np.rint(query_vecs[0]).astype(np.int32)
    top_k_idxes = zk_ivf_pq_query(
        query,
        top_k,
        labels,
        quant_center,
        quant_code_books,
        quant_vecs,
        id_groups,
    )
