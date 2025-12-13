import numpy as np
from sklearn.cluster import KMeans  # 或 MiniBatchKMeans
import faiss
from typing import Dict, Tuple, Optional


def kmeans_with_ids(
    X: np.ndarray,
    k: int,
    niter: int = 10,
    random_state: Optional[int] = 0,
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
    ids = np.arange(N, dtype=np.int64)

    km = KMeans(n_clusters=k, random_state=random_state, max_iter=niter)
    labels = km.fit_predict(X).astype(np.int64)  # (N,)
    centers = km.cluster_centers_  # (k, D)
    centers = np.rint(centers).astype(np.int64)  # 转64

    # 将每个簇对应的样本 id 收集起来
    id_groups = {c: ids[labels == c] for c in range(k)}
    return centers, id_groups, labels


def faiss_kmeans_with_ids(
    X: np.ndarray,
    k: int,
    niter: int = 25,
    random_state: Optional[int] = 1234,
) -> Tuple[np.ndarray, Dict[int, np.ndarray], np.ndarray]:
    """
    使用 FAISS 做 KMeans 并返回簇中心与每簇的样本 id。
    """
    assert X.ndim == 2
    N, D = X.shape
    ids = np.arange(N, dtype=np.int64)

    # FAISS 通常使用 float32
    X32 = X.astype(np.float32, copy=False)

    # 训练 KMeans
    kmeans = faiss.Kmeans(d=D, k=k, niter=niter, verbose=True, gpu=1)
    # res = faiss.StandardGpuResources()
    # index = faiss.GpuIndexFlatL2(res, D)
    # kmeans = faiss.Clustering(D, k)
    # kmeans.niter = niter
    # kmeans.verbose = True
    if random_state is not None:
        # faiss Kmeans 使用 seed 控制初始化随机性
        kmeans.seed = int(random_state)
    kmeans.train(X32)  # 得到 centroids
    faiss.gpu_sync_all_devices()
    centers = kmeans.centroids  # (k, D), float32

    # 用中心建一个Index，把每个样本分到最近的中心（1-NN 到中心）
    index = faiss.IndexFlatL2(D)

    index.add(centers)  # 加入 k 个中心
    Dists, I = index.search(X32, 1)  # 每个样本最近中心的索引
    labels = I.ravel().astype(np.int32)  # (N,)

    id_groups = {c: ids[labels == c] for c in range(k)}
    return centers, id_groups, labels
