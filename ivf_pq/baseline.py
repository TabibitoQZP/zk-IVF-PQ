import math
import numpy as np
import faiss


def ivfpq_search(
    xb: np.ndarray,
    xq: np.ndarray,
    top_k: int = 10,
    *,
    metric: str = "l2",  # "l2" | "ip" | "cosine"
    nlist: int | None = None,  # 粗聚类簇数；默认 ~ 4*sqrt(N)
    M: int = 16,  # PQ 子空间数，要求 D % M == 0
    nbits: int = 8,  # 每子空间码字比特数（256 码字）
    nprobe: int = 16,  # 查询探测簇数
    train_size: int | None = None,  # 训练样本数（默认 min(100k, N)）
    use_gpu: bool = False,  # 若可用则搬到 GPU
    precompute_lut: bool = True,  # 预计算查找表，加速大 nprobe
    use_opq: bool = False,  # 使用 OPQ 提升精度
    return_distances: bool = False,  # 是否同时返回距离/相似度
    random_state: int | None = 0,  # 随机种子（用于训练采样）
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    构建 IVF-PQ 并对 xq 检索。

    Returns
    -------
    I : (n, top_k) int32
        每行是该查询向量的 top-k 索引，排序为：
        - metric="l2"：距离从小到大（等价“相似度高→低”）
        - metric="ip"/"cosine"：相似度从大到小
    或 (I, D/S) ：若 return_distances=True，同时返回距离/相似度矩阵（float32）
    """
    # -------- 输入校验与预处理 --------
    xb = np.ascontiguousarray(xb, dtype="float32")
    xq = np.ascontiguousarray(xq, dtype="float32")
    N, D = xb.shape
    n, Dq = xq.shape
    if D != Dq:
        raise ValueError(f"D mismatch: xb is {D}, xq is {Dq}")
    if D % M != 0:
        raise ValueError(f"D({D}) must be divisible by M({M}) for PQ.")
    if top_k <= 0 or top_k > max(1, N):
        raise ValueError("top_k must be in [1, N].")

    # 缺省 nlist：4*sqrt(N)，再做边界与训练样本的安全裁剪
    if nlist is None:
        nlist = max(2, int(4 * math.sqrt(N)))

    # 训练样本数
    if train_size is None:
        train_size = min(100_000, N)
    train_size = max(2 * M, min(train_size, N))  # 至少给 PQ 一点余量

    # 保证 nlist 不超过训练样本数
    if nlist > train_size:
        nlist = max(2, train_size)

    # 选择度量与（可选）归一化
    use_ip_metric = False
    xb_in, xq_in = xb, xq
    if metric.lower() == "cosine":
        xb_in = xb.copy()
        xq_in = xq.copy()
        faiss.normalize_L2(xb_in)
        faiss.normalize_L2(xq_in)
        use_ip_metric = True
    elif metric.lower() == "ip":
        use_ip_metric = True
    elif metric.lower() == "l2":
        use_ip_metric = False
    else:
        raise ValueError("metric must be one of {'l2','ip','cosine'}")

    # -------- 构建量化器 + IVF-PQ（可选 OPQ）--------
    if use_ip_metric:
        quantizer = faiss.IndexFlatIP(D)
        metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        quantizer = faiss.IndexFlatL2(D)
        metric_type = faiss.METRIC_L2

    ivfpq = faiss.IndexIVFPQ(quantizer, D, nlist, M, nbits, metric_type)
    ivfpq.use_precomputed_table = bool(precompute_lut)

    if use_opq:
        opq = faiss.OPQMatrix(D, M)
        opq.niter = 20
        index_cpu = faiss.IndexPreTransform(opq, ivfpq)
    else:
        index_cpu = ivfpq

    # --------（可选）搬到 GPU；失败则回退 CPU --------
    index = index_cpu
    res = None
    if use_gpu and hasattr(faiss, "StandardGpuResources"):
        try:
            res = faiss.StandardGpuResources()
            # 说明：IndexPreTransform 也可尝试搬到 GPU；若失败自动回退
            index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        except Exception as e:
            print(f"[ivfpq_search] GPU not available, fallback to CPU: {e}")
            index = index_cpu
    elif use_gpu:
        print("[ivfpq_search] faiss-gpu not detected, using CPU.")

    # -------- 训练（必须先于 add）--------
    rng = np.random.default_rng(seed=random_state)
    train_idx = rng.choice(N, size=train_size, replace=False)
    xt = np.ascontiguousarray(xb_in[train_idx])

    index.train(xt)

    # -------- 建库与检索 --------
    index.add(xb_in)
    index.nprobe = int(nprobe)

    # 注意：L2 返回距离（小→大），IP/余弦返回相似度（大→小）
    D_or_S, I = index.search(xq_in, top_k)

    if return_distances:
        return I.astype("int32"), D_or_S
    return I.astype("int32")
