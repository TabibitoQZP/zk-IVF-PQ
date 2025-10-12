import numpy as np
from vec_data_load.sift import SIFT

from ivf_pq.baseline import ivfpq_search


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

    res = ivfpq_search(vecs, query_vecs, top_k, nlist=128)
    print(res)
