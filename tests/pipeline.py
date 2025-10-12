import numpy as np
from vec_data_load.sift import SIFT

from ivf_pq.pipeline import ivf_pq_learn, ivf_pq_query


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
    labels, quant_center, quant_code_books, quant_vecs, id_groups = ivf_pq_learn(
        vecs,
        # n_list=4096,
        # n_iter=16,
    )

    print("IVF-PQ learning finished.")

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
