import numpy as np


def rebalance_clusters(
    vecs: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
    cluster_bound: int,
):
    """
    在给定 cluster_bound 的前提下, 重新分配样本到已有簇中心, 使得每个簇大小
    不超过 cluster_bound。只做簇间重新分配, 不改变 centers。

    返回:
        new_labels:  重排后的标签, shape (N,)
        id_groups:   {cluster_id: np.ndarray[样本下标]}
        changed_count: 标签发生变化的样本数量
    """
    if cluster_bound is None:
        raise ValueError("cluster_bound 不能为空")
    if cluster_bound <= 0:
        raise ValueError("cluster_bound 必须为正整数")

    vecs = np.asarray(vecs)
    centers = np.asarray(centers)
    labels = np.asarray(labels).astype(np.int64, copy=False)

    N, D = vecs.shape
    n_list = centers.shape[0]

    if labels.shape[0] != N:
        raise ValueError("labels 与 vecs 的样本数不一致")

    if n_list * cluster_bound < N:
        raise ValueError(
            f"无法在 cluster_bound={cluster_bound} 下重排: "
            f"总容量 {n_list * cluster_bound} 小于样本数 {N}"
        )

    original_labels = labels.copy()
    new_labels = labels.copy()

    # 主循环: 直到不存在超载簇为止
    while True:
        sizes = np.bincount(new_labels, minlength=n_list)

        overfull = np.where(sizes > cluster_bound)[0]
        print(overfull)
        if overfull.size == 0:
            break

        # 严格未满簇才能作为目标簇
        free_clusters = np.where(sizes < cluster_bound)[0]
        if free_clusters.size == 0:
            # 理论上在上面的容量检查通过时不应发生, 这里防御性报错
            raise RuntimeError("存在超载簇但没有可用的目标簇, 重排失败")

        # 预先计算 free_clusters 的中心, 用 float64 计算距离更安全
        centers_free = centers[free_clusters].astype(np.float64, copy=False)
        # 每个 free cluster 当前还能接收多少样本
        free_capacity = cluster_bound - sizes[free_clusters]
        capacity_map = {
            int(c): int(cap)
            for c, cap in zip(free_clusters.tolist(), free_capacity.tolist())
        }

        # 收集所有候选移动: (样本 idx, 源簇, 目标簇, 代价)
        cand_indices = []
        cand_src = []
        cand_tgt = []
        cand_cost = []

        for c in overfull:
            idx_in_c = np.where(new_labels == c)[0]
            if idx_in_c.size == 0:
                continue

            # 当前簇中心
            center_c = centers[c].astype(np.float64, copy=False)

            # 只计算这些样本到当前簇与所有 free 簇中心的距离
            X = vecs[idx_in_c].astype(np.float64, copy=False)  # (Nc, D)

            diff_cur = X - center_c
            dist_cur = np.sum(diff_cur * diff_cur, axis=1)  # (Nc,)

            # 到 free_clusters 的距离: (Nc, |C_free|)
            diff_free = X[:, None, :] - centers_free[None, :, :]
            dist_free = np.sum(diff_free * diff_free, axis=2)

            # 为每个样本选择当前 free 集合中的最近簇
            nearest_pos = np.argmin(dist_free, axis=1)  # 在 free_clusters 中的位置
            nearest_clusters = free_clusters[nearest_pos]  # 映射成真实簇 id
            dist_alt = dist_free[np.arange(idx_in_c.size), nearest_pos]

            cost = dist_alt - dist_cur  # (Nc,)

            cand_indices.append(idx_in_c)
            cand_src.append(np.full(idx_in_c.shape, c, dtype=np.int64))
            cand_tgt.append(nearest_clusters.astype(np.int64))
            cand_cost.append(cost.astype(np.float64))

        if not cand_indices:
            # 没有可移动的候选 (例如所有 free 簇容量为 0), 防御性退出
            raise RuntimeError("没有可行的迁移候选, 重排失败")

        cand_indices = np.concatenate(cand_indices)
        cand_src = np.concatenate(cand_src)
        cand_tgt = np.concatenate(cand_tgt)
        cand_cost = np.concatenate(cand_cost)

        # 按代价从小到大尝试迁移
        order = np.argsort(cand_cost)
        moved_in_this_round = 0

        for k in order:
            i = int(cand_indices[k])
            c_src = int(cand_src[k])
            c_tgt = int(cand_tgt[k])

            # 源簇已经不再超载, 就没必要继续从里面迁出
            if sizes[c_src] <= cluster_bound:
                continue

            # 目标簇是否还有容量
            if capacity_map.get(c_tgt, 0) <= 0:
                continue

            if new_labels[i] != c_src:
                # 该样本在本轮中可能已被迁移过
                continue

            # 执行迁移
            new_labels[i] = c_tgt
            sizes[c_src] -= 1
            sizes[c_tgt] += 1
            capacity_map[c_tgt] -= 1
            moved_in_this_round += 1

        if moved_in_this_round == 0:
            # 本轮没有任何样本被迁移, 但仍然存在超载簇, 说明重排陷入僵局
            raise RuntimeError("重排过程中未能进一步减少超载簇, 重排失败")

    # 构建新的 id_groups
    id_groups = {c: [] for c in range(n_list)}
    for idx, c in enumerate(new_labels):
        id_groups[int(c)].append(int(idx))

    for c in id_groups:
        id_groups[c] = np.array(id_groups[c], dtype=np.int64)

    changed_count = int(np.sum(new_labels != original_labels))
    return new_labels, id_groups, changed_count
