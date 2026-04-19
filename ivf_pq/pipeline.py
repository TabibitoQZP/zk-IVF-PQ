import numpy as np
from ivf_pq.layout import apply_layout
from ivf_pq.util.kmeans import faiss_kmeans_with_ids


def ivf_pq_learn(
    vecs: np.ndarray,
    n_list=64,
    n_iter=64,
    M=8,
    K=256,
    layout: str | None = None,
):
    vecs = apply_layout(np.asarray(vecs), layout)
    N, D = vecs.shape

    d = D // M
    center, id_groups, labels = faiss_kmeans_with_ids(vecs, n_list, n_iter)

    centers = center[labels]

    res_vecs = vecs - centers

    code_books = []
    quant_vecs = []
    for i in range(0, D, d):
        res_slide = res_vecs[:, i : i + d]
        slide_center, slide_id_groups, slide_labels = faiss_kmeans_with_ids(
            res_slide, K, n_iter
        )
        code_books.append(slide_center)
        quant_vecs.append(slide_labels)

    code_books = np.array(code_books)
    quant_vecs = np.ascontiguousarray(np.array(quant_vecs).T)

    quant_code_books = np.rint(code_books).astype(np.int32)
    quant_center = np.rint(center).astype(np.int32)

    # (N,), (n_list,D), (M,K,D/M), (N,M)
    return labels, quant_center, quant_code_books, quant_vecs, id_groups


def ivf_pq_query(
    query: np.ndarray,
    top_k: int,
    labels: np.ndarray,
    quant_center: np.ndarray,
    quant_code_books: np.ndarray,
    quant_vecs: np.ndarray,
    id_groups: dict,
    n_probe: int = 8,
    layout: str | None = None,
):
    query = apply_layout(np.asarray(query), layout)
    # N = labels.shape[0]

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
