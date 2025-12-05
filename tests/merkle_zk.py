import numpy as np

from ivf_pq.merkle_zk import ivf_pq_learn, zk_ivf_pq_query
from vec_data_load.sift import SIFT


def brute_force_knn(
    base_vecs: np.ndarray, query_vec: np.ndarray, top_k: int
) -> np.ndarray:
    """
    使用暴力枚举的方式，以 L2 距离计算 KNN 结果，作为 ground truth。
    """
    base_vecs = np.asarray(base_vecs)
    query_vec = np.asarray(query_vec, dtype=base_vecs.dtype)

    diff = base_vecs - query_vec
    dist2 = np.sum(diff * diff, axis=1)
    topk_idx = np.argsort(dist2)[:top_k]
    return topk_idx.astype(np.int64)


if __name__ == "__main__":
    n_list = 16
    n_probe = 4
    top_k = 64
    data_root = "data/siftsmall/"
    # data_root = "data/sift/"
    sift = SIFT(data_root)

    base_vecs = sift.base_vecs  # (N, D)
    query_vecs = sift.query_vecs  # (100, D)
    gt_vecs = sift.gt_vecs  # (100, 100)

    print("IVF-PQ (Merkle ZK) learning start.")
    labels, center, code_books, quant_vecs, id_groups = ivf_pq_learn(
        base_vecs,
        n_list=n_list,
        # n_list=4096,
        # n_iter=16,
    )
    print("IVF-PQ (Merkle ZK) learning finished.")

    # 这里只测试一条查询路径，主要验证证明系统跑通
    for i in range(8):
        # 用原始浮点向量做 L2 暴力检索，作为 ground truth
        brute_top_k_idxes = brute_force_knn(base_vecs, query_vecs[i], top_k)

        # ZK 版本仍然使用 int64 查询
        query = np.rint(query_vecs[i]).astype(np.int64)
        zk_top_k_idxes, _ = zk_ivf_pq_query(
            query,
            center,
            code_books,
            quant_vecs,
            id_groups,
            top_k=top_k,
        )

        # 以暴力 KNN 结果作为 ground truth，计算 pass@k
        intersect = np.intersect1d(zk_top_k_idxes, brute_top_k_idxes)
        pass_at_k = len(intersect) / float(top_k)

        print("#" * 32, i, "#" * 32)
        print("ZK Merkle IVF-PQ top-k indices:", zk_top_k_idxes)
        print("Brute-force L2 top-k indices (ground truth):", brute_top_k_idxes)
        print("Original dataset gt indices (for reference):", gt_vecs[i][:top_k])
        print(f"ZK Merkle IVF-PQ pass@{top_k}: {pass_at_k:.4f}")
