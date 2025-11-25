import numpy as np
from itertools import combinations
from typing import List, Tuple
from scipy.optimize import linprog


def naive_upperbound(
    clusters: List[Tuple[float, np.ndarray]], n_probe: int = 8
) -> float:
    """
    朴素上界：直接把最大的 n_probe 个 cluster size 相加。
    不考虑几何结构，只是一个通用的 worst-case 上界。
    """
    sizes = np.array([c[0] for c in clusters], dtype=float)
    if n_probe >= len(sizes):
        return float(sizes.sum())
    # 按 size 降序取前 n_probe 个
    sizes_sorted = np.sort(sizes)[::-1]
    return float(sizes_sorted[:n_probe].sum())


def _is_knn_set_feasible(
    centers: np.ndarray, S: Tuple[int, ...], tol: float = 1e-9
) -> bool:
    """
    检查给定下标集合 S 是否可以成为某个查询点 q 的“最近 n_probe 个中心”集合。
    使用线性规划检查多面体是否非空。

    centers: shape = (K, d)
    S: 一个包含若干中心下标的 tuple
    """
    K, d = centers.shape
    S_set = set(S)
    others = [j for j in range(K) if j not in S_set]

    # 如果所有中心都在 S 里，那显然可以（任意 q 都行）
    if not others:
        return True

    A_ub = []
    b_ub = []

    # 对所有 i in S, j not in S 构造约束：
    # 2 (c_j - c_i) · q <= ||c_j||^2 - ||c_i||^2
    norms2 = np.sum(centers**2, axis=1)

    for i in S:
        ci = centers[i]
        norm_ci2 = norms2[i]
        for j in others:
            cj = centers[j]
            norm_cj2 = norms2[j]
            A_ub.append(2.0 * (cj - ci))
            b_ub.append(norm_cj2 - norm_ci2)

    A_ub = np.array(A_ub, dtype=float)
    b_ub = np.array(b_ub, dtype=float)

    # 目标函数无所谓，用零向量即可
    c = np.zeros(d, dtype=float)

    # 变量 q 没有边界（(-inf, +inf)^d）
    bounds = [(None, None)] * d

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    # res.success 为 True 表示存在可行解
    return res.success


def geometry_aware_upperbound(
    clusters: List[Tuple[float, np.ndarray]], n_probe: int = 8
) -> float:
    """
    几何约束下的最紧上界（理论上是 exact worst-case probing cost）。

    输入:
        clusters: List[(cluster_size, center_vector)], center_vector 可以是 list/tuple/np.ndarray
        n_probe: 需要探测的 cluster 数量

    输出:
        一个 float，上界值（在有限精度下其实就是 sup_q cost(q)）。

    ⚠ 复杂度为 C(K, n_probe)，仅适合 K 较小时用于研究/论文示例。
    """
    if len(clusters) == 0:
        return 0.0

    K = len(clusters)
    if n_probe >= K:
        # 所有 cluster 都要扫一遍
        return float(sum(c[0] for c in clusters))

    # 拆出 sizes 和 centers，并把 centers 转成 np.ndarray
    sizes = np.array([c[0] for c in clusters], dtype=float)
    centers = np.array([np.asarray(c[1], dtype=float) for c in clusters])

    # 按 size 降序排序，把大的放前面，有利于更快找到 worst-case 组合
    order = np.argsort(-sizes)  # descending
    sizes_sorted = sizes[order]
    centers_sorted = centers[order]

    # 先生成所有组合（在排序后的索引中），并记录它们的 size 和
    # combos: List[(neg_total_size, tuple_of_indices_in_sorted_space)]
    combos = []
    for idx_tuple in combinations(range(K), n_probe):
        total = float(sizes_sorted[list(idx_tuple)].sum())
        combos.append((-total, idx_tuple))

    # 按总 size 从大到小排序（负号是为了用升序 sort）
    combos.sort()

    # 依次尝试每个组合：第一个几何可行的组合就是 worst-case
    for neg_total, idx_tuple_in_sorted in combos:
        total = -neg_total

        # 把“排序后的索引”映射回原始索引只是为了 debug/解释，
        # 实际上可行性只依赖 centers，所以直接在 centers_sorted 上做即可。
        # orig_indices = tuple(order[list(idx_tuple_in_sorted)])  # 若想保留原 index，可以用这个

        if _is_knn_set_feasible(centers_sorted, idx_tuple_in_sorted):
            # 找到了几何上可实现的、size 和最大的组合
            return total

    # 理论上不应走到这里（总有一些组合是可行的），
    # 为了安全起见，fallback 到朴素上界。
    return naive_upperbound(clusters, n_probe=n_probe)


if __name__ == "__main__":
    clusters = []
    for _ in range(256):
        curr_size = np.random.randint(0, 101)
        vec = np.random.randint(0, 16384, size=1024)
        clusters.append((curr_size, vec))
    # clusters = [
    #     (100, [0.0, 0.0]),
    #     (80, [1.0, 0.0]),
    #     (50, [0.0, 1.0]),
    #     (30, [10.0, 0.0]),
    #     (20, [0.0, 10.0]),
    #     (10, [5.0, 5.0]),
    # ]

    print("naive:", naive_upperbound(clusters, n_probe=3))
    print("geom :", geometry_aware_upperbound(clusters, n_probe=3))
