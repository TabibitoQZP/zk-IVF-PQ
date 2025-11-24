import numpy as np
from vec_data_load.sift import SIFT

from ivf_pq.pipeline import ivf_pq_learn

from zk_IVF_PQ.zk_IVF_PQ import py_ivf_pq_verify_proof


# TODO: 修一下, 适应zk系统
def ivf_pq_query(
    query: np.ndarray,
    quant_center: np.ndarray,
    quant_code_books: np.ndarray,
    quant_vecs: np.ndarray,
    id_groups: dict,
    n_probe: int = 8,
):
    # 计算距离
    diff = quant_center - query
    dis_2 = np.einsum("ij,ij->i", diff, diff)
    cluster_idx = np.argsort(dis_2, kind="stable")
    # sorted_dis_2 = dis_2[cluster_idx]

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


def iou_set(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 and b.size == 0:
        return 1.0
    inter = np.intersect1d(a, b)
    union = np.union1d(a, b)
    return inter.size / union.size


if __name__ == "__main__":
    top_k = 10
    data_root = "data/siftsmall/"
    # data_root = "data/sift/"
    sift = SIFT(data_root)

    vecs = sift.base_vecs  # (N,D)
    query_vecs = sift.query_vecs  # (100,D)
    gt_vecs = sift.gt_vecs  # (100,100)

    print("IVF-PQ learning start.")

    (
        labels,
        ivf_centers,
        codebooks,
        quant_vecs,
        id_groups,
    ) = ivf_pq_learn(
        vecs,
        # n_list=4096,
        # n_iter=16,
    )

    print("IVF-PQ learning finished.")

    exit(0)

    # TODO: 计算所需的内容
    query = None
    sorted_idx_dis = None
    filtered_centers = None
    probe_count = None
    filtered_vecs = None
    vecs_cluster_hot = None
    result = py_ivf_pq_verify_proof(
        ivf_centers,
        query,
        sorted_idx_dis,
        filtered_centers,
        probe_count,
        filtered_vecs,
        vecs_cluster_hot,
        codebooks,
    )

    iou_list = []
    for i in range(query_vecs.shape[0]):
        query = np.rint(query_vecs[i]).astype(np.int32)
        top_k_idxes = ivf_pq_query(
            query,
            top_k,
            labels,
            quant_center,
            quant_code_books,
            quant_vecs,
            id_groups,
        )

        print("#" * 32, i, "#" * 32)
        print(top_k_idxes)
        print(gt_vecs[i][:top_k])
        iou = iou_set(top_k_idxes, gt_vecs[i][:top_k])
        print(iou)
        iou_list.append(iou)

    print(iou_list)
    print(sum(iou_list) / len(iou_list))
