import numpy as np
from ivf_pq.zk import ivf_pq_learn, zk_ivf_pq_query
from vec_data_load.sift import SIFT

if __name__ == "__main__":
    top_k = 10
    data_root = "data/siftsmall/"
    # data_root = "data/sift/"
    sift = SIFT(data_root)

    vecs = sift.base_vecs  # (N,D)
    query_vecs = sift.query_vecs  # (100,D)
    gt_vecs = sift.gt_vecs  # (100,100)

    print("IVF-PQ learning start.")
    labels, center, code_books, quant_vecs, id_groups = ivf_pq_learn(
        vecs,
        # n_list=4096,
        # n_iter=16,
    )

    print("IVF-PQ learning finished.")

    # for i in range(query_vecs.shape[0]):
    for i in range(1):
        query = np.rint(query_vecs[i]).astype(np.int32)
        top_k_idxes = zk_ivf_pq_query(query, center, code_books, quant_vecs, id_groups)
