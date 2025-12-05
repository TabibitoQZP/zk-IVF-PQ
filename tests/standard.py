import numpy as np

from ivf_pq.standard import ivf_pq_learn, ivf_pq_query
from vec_data_load.sift import SIFT


def recall_at_k(pred, gt, k: int) -> float:
    pred = np.asarray(pred)[:k]
    gt = np.asarray(gt)[:k]
    if pred.size == 0 or gt.size == 0:
        return 0.0
    inter = np.intersect1d(pred, gt)
    return float(inter.size) / float(k)


if __name__ == "__main__":
    top_k = 10
    data_root = "data/siftsmall/"
    # data_root = "data/sift/"
    sift = SIFT(data_root)

    base_vecs = sift.base_vecs  # (N, D)
    query_vecs = sift.query_vecs  # (Q, D)
    gt_vecs = sift.gt_vecs  # (Q, 100)

    print("Standard IVF-PQ (float) learning start.")
    labels, center, code_books, quant_vecs, id_groups = ivf_pq_learn(
        base_vecs,
        # n_list=4096,
        # n_iter=16,
    )
    print("Standard IVF-PQ (float) learning finished.")

    recalls = []
    for i in range(query_vecs.shape[0]):
        query = np.asarray(query_vecs[i], dtype=np.float32)
        top_k_idxes = ivf_pq_query(
            query,
            top_k,
            labels,
            center,
            code_books,
            quant_vecs,
            id_groups,
        )

        gt_top_k = gt_vecs[i][:top_k]
        r_at_k = recall_at_k(top_k_idxes, gt_top_k, top_k)

        print("#" * 32, i, "#" * 32)
        print("Standard IVF-PQ top-k indices:", top_k_idxes)
        print("Ground-truth top-k indices:    ", gt_top_k)
        print(f"Recall@{top_k}: {r_at_k:.4f}")
        recalls.append(r_at_k)

    if recalls:
        mean_recall = float(np.mean(recalls))
        print("Per-query Recall@{}:".format(top_k), recalls)
        print("Average Recall@{}: {:.4f}".format(top_k, mean_recall))
