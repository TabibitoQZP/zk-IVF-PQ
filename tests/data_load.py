from ivf_pq.util.kmeans import faiss_kmeans_with_ids
from vec_data_load.sift import SIFT

if __name__ == "__main__":
    data_root = "data/siftsmall/"
    sift = SIFT(data_root)

    center, id_groups, labels = faiss_kmeans_with_ids(sift.base_vecs, 128, 64)

    sz = []
    for k, v in id_groups.items():
        sz.append(v.shape[0])
    print(min(sz))
    print(max(sz))
