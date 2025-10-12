import numpy as np
from sklearn.cluster import KMeans  # 或 MiniBatchKMeans
import faiss
from typing import Dict, Tuple


def kmeans_with_ids(
    X: np.ndarray, k: int, random_state: int = 0, n_init: int = 10
) -> Tuple[np.ndarray, Dict[int, np.ndarray], np.ndarray]:
    """
    X: shape (N, D) 的 ndarray
    k: 簇数
    返回:
      centers: (k, D) 聚类中心
      id_groups: {cluster_label: np.array([...ids...])}
      labels: (N,) 每个样本的簇标签
    """
    assert X.ndim == 2, "X 应为二维 [N, D]"
    N = X.shape[0]
    ids = np.arange(N, dtype=np.int32)

    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X)  # (N,)
    centers = km.cluster_centers_  # (k, D)

    # 将每个簇对应的样本 id 收集起来
    id_groups = {c: ids[labels == c] for c in range(k)}
    return centers, id_groups, labels


def faiss_kmeans_with_ids(
    X: np.ndarray, k: int, niter: int = 25, gpu: bool = False
) -> Tuple[np.ndarray, Dict[int, np.ndarray], np.ndarray]:
    """
    使用 FAISS 做 KMeans 并返回簇中心与每簇的样本 id。
    """
    assert X.ndim == 2
    N, D = X.shape
    ids = np.arange(N, dtype=np.int32)

    # FAISS 通常使用 float32
    X32 = X.astype(np.float32, copy=False)

    # 训练 KMeans
    kmeans = faiss.Kmeans(d=D, k=k, niter=niter, verbose=False, gpu=gpu)
    kmeans.train(X32)  # 得到 centroids
    centers = kmeans.centroids  # (k, D), float32

    # 用中心建一个Index，把每个样本分到最近的中心（1-NN 到中心）
    index = faiss.IndexFlatL2(D)
    if gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(centers)  # 加入 k 个中心
    Dists, I = index.search(X32, 1)  # 每个样本最近中心的索引
    labels = I.ravel().astype(np.int32)  # (N,)

    id_groups = {c: ids[labels == c] for c in range(k)}
    return centers, id_groups, labels
