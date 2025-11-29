import numpy as np

from ivf_pq.merkle_zk import ivf_pq_learn, zk_ivf_pq_query
from vec_data_load.sift import SIFT


if __name__ == "__main__":
    top_k = 10
    data_root = "data/siftsmall/"
    # data_root = "data/sift/"
    sift = SIFT(data_root)

    base_vecs = sift.base_vecs  # (N, D)
    query_vecs = sift.query_vecs  # (100, D)
    gt_vecs = sift.gt_vecs  # (100, 100)

    print("IVF-PQ (Merkle ZK) learning start.")
    labels, center, code_books, quant_vecs, id_groups = ivf_pq_learn(
        base_vecs,
        # n_list=4096,
        # n_iter=16,
    )
    print("IVF-PQ (Merkle ZK) learning finished.")

    # 这里只测试一条查询路径，主要验证证明系统跑通
    for i in range(1):
        query = np.rint(query_vecs[i]).astype(np.int64)
        top_k_idxes = zk_ivf_pq_query(
            query,
            center,
            code_books,
            quant_vecs,
            id_groups,
            top_k=top_k,
        )

        print("#" * 32, i, "#" * 32)
        print("ZK Merkle IVF-PQ top-k indices:", top_k_idxes)
        print("Ground-truth top-k indices:", gt_vecs[i][:top_k])

