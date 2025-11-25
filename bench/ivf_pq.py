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
    # 计算上界
    ub = upperbound(id_groups)

    # 计算距离
    diff = quant_center - query
    dis_2 = np.einsum("ij,ij->i", diff, diff)
    cluster_idx = np.argsort(dis_2, kind="stable")
    sorted_idx_dis = np.stack([cluster_idx, dis_2[cluster_idx]], axis=0)
    filtered_centers = quant_center[cluster_idx[:n_probe]]
    probe_count = np.array([len(id_groups[i]) for i in cluster_idx], dtype=np.uint32)

    all_cluster_idxes = []
    all_vec_idxes = []
    all_dis_2 = []
    # TODO: 筛选出filtered_vecs, vecs_cluster_hot
    second_idxes = []
    all_vecs = []
    for si, i in enumerate(cluster_idx[:n_probe]):
        curr_center = quant_center[i]
        res_query = query - curr_center
        ids = id_groups[i]

        curr_N = ids.shape[0]

        curr_vecs = quant_vecs[ids]
        all_vecs.append(curr_vecs)
        curr_q_vecs = quant_code_books[
            np.arange(curr_vecs.shape[1])[None, :], curr_vecs, :
        ].reshape(curr_vecs.shape[0], -1)

        curr_diff = curr_q_vecs - res_query
        curr_dis_2 = np.einsum("ij,ij->i", curr_diff, curr_diff)

        all_cluster_idxes.append(np.full((curr_N,), i, dtype=np.int32))
        all_vec_idxes.append(ids)
        all_dis_2.append(curr_dis_2)

        second_idxes.append(si)
    print(second_idxes)

    all_cluster_idxes = np.concatenate(all_cluster_idxes)
    all_vec_idxes = np.concatenate(all_vec_idxes)
    all_dis_2 = np.concatenate(all_dis_2)
    second_idxes = np.concatenate(second_idxes)
    all_vecs = np.concatenate(all_vecs)
    all_sorted_idx = np.argsort(all_dis_2, kind="stable")

    all_cluster_idxes = all_cluster_idxes[all_sorted_idx]
    all_vec_idxes = all_vec_idxes[all_sorted_idx]
    all_dis_2 = all_dis_2[all_sorted_idx]
    second_idxes = second_idxes[all_sorted_idx]
    all_vecs = all_vecs[all_sorted_idx]
    # 根据second_idxes来构建one-hot
    origin_one_hot = np.zeros([len(second_idxes), n_probe], dtype=np.uint32)
    origin_one_hot = origin_one_hot[np.arange(len(second_idxes)), second_idxes]
    vecs_cluster_hot = np.zeros([ub, n_probe], dtype=np.uint32)
    vecs_cluster_hot[: len(second_idxes), :] = origin_one_hot
    # 补全filtered_vecs
    filtered_vecs = np.zeros([ub, len(all_vecs[0])], dtype=np.uint32)
    filtered_vecs[: len(all_vecs), :] = all_vecs

    result = py_ivf_pq_verify_proof(
        quant_center,
        query,
        sorted_idx_dis,
        filtered_centers,
        probe_count,
        filtered_vecs,
        vecs_cluster_hot,
        codebooks,
    )
    print(result)

    return all_vec_idxes[:top_k]


def iou_set(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 and b.size == 0:
        return 1.0
    inter = np.intersect1d(a, b)
    union = np.union1d(a, b)
    return inter.size / union.size


# 目前只能取一个非常松的上界
def upperbound(id_groups, n_probe=8):
    size = [v.shape[0] for v in id_groups.values()]
    return sum(size[-n_probe:])


if __name__ == "__main__":
    top_k = 10
    data_root = "data/siftsmall/"
    # data_root = "data/sift/"
    sift = SIFT(data_root)

    vecs = sift.base_vecs  # (N,D)
    query_vecs = sift.query_vecs  # (100,D)
    gt_vecs = sift.gt_vecs  # (100,100)
    print(vecs.shape)

    print("IVF-PQ learning start.")

    (
        labels,
        ivf_centers,
        codebooks,
        quant_vecs,
        id_groups,  # id分配
    ) = ivf_pq_learn(
        vecs,
        # n_list=4096,
        # n_iter=16,
    )
    print(quant_vecs.shape)

    print("IVF-PQ learning finished.")

    for i in range(query_vecs.shape[0]):
        query = np.rint(query_vecs[i]).astype(np.int32)
        top_k_idxes = ivf_pq_query(
            query,
            ivf_centers,
            codebooks,
            quant_vecs,
            id_groups,
        )
        break

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
