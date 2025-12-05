import numpy as np
from zk_IVF_PQ.zk_IVF_PQ import py_set_based_with_merkle, py_set_based_without_merkle

from bench import MAX_DIS, data_gen

"""
函数签名

pub fn set_based_ivf_pq_proof(
    query: Vec<i64>,               // 查询向量 (D,)
    ivf_center: Vec<Vec<i64>>,     // ivf簇中心 (n_list,D)
    vpqss: Vec<Vec<Vec<i64>>>,     // 这里给原始向量, 手动改one-hot (n_probe,n,M)
    valids: Vec<Vec<i64>>,         // vpqss中向量是否valid (n_probe,n)
    itemss: Vec<Vec<i64>>,         // vpqss中向量对应的查询量 (n_probe,n)
    codebooks: Vec<Vec<Vec<i64>>>, // 全局码本 (M,K,d)
    ivf_roots: Vec<u64>,           // 这里给一下ivf各个root, 用来手算和还原数据 (n_list,)
    top_k: i64,                    // 明确取哪top_k
    // 后面的可以在rust内部算, 也可以python端算完传入, 这里用传入实现, 懒得写了...
    cluster_idx_dis: Vec<Vec<i64>>,        // (n_list,2)
    ordered_vpqss_item_dis: Vec<Vec<i64>>, // vpqss中计算的距离和item集合 (n_probe*n,2)
) -> Result<(f64, f64, f64, u64, u64), Box<dyn std::error::Error>>
"""


def bench(D, n_list, M, K, d, n_probe, n, top_k=64, merkled=True):
    (
        query,
        ivf_center,
        cluster_idxes,  # 这个不需要了, 用cluster_idx_dis可以推
        vpqss,
        valids,
        itemss,
        codebooks,  # (M,K,d)
        ivf_roots,  # 这个是uint64, 其他都是int64
    ) = data_gen(D, n_list, M, K, d, n_probe, n)

    # 手写完成对cluster_idx_dis的计算
    c = np.sum((ivf_center - query) ** 2, axis=1).astype(np.int64)
    order = np.argsort(c, kind="stable")
    order = order.astype(np.int64)
    cluster_idx_dis = np.column_stack((order, c[order]))

    for i in range(0, n_probe):
        assert cluster_idxes[i] == cluster_idx_dis[i, 0], (
            "Something wrong calused not equal"
        )

    # TODO: 手写完成对ordered_vpqss_item_dis的计算, 难写一点
    ordered_vpqss_item_dis = []
    for i in range(0, n_probe):
        # 注意这里的要挑部分
        delta_query = query - ivf_center[cluster_idxes[i]]
        for j in range(0, n):
            curr_vpq = vpqss[i][j]
            selected = codebooks[np.arange(M), curr_vpq]
            x = selected.reshape(M * selected.shape[1])
            curr_diff = delta_query - x
            curr_dis = np.dot(curr_diff, curr_diff)
            if valids[i, j] == 0:
                curr_dis = MAX_DIS
            ordered_vpqss_item_dis.append([itemss[i][j], curr_dis])
    ordered_vpqss_item_dis = np.array(ordered_vpqss_item_dis, dtype=np.int64)
    idx = np.argsort(ordered_vpqss_item_dis[:, 1], kind="stable")
    ordered_vpqss_item_dis = ordered_vpqss_item_dis[idx]
    if merkled: 
        result = py_set_based_with_merkle(
        query,
        ivf_center,
        vpqss,
        valids,
        itemss,
        codebooks,
        ivf_roots,
        top_k,
        cluster_idx_dis,
        ordered_vpqss_item_dis,
    )
    else:
        result = py_set_based_without_merkle(
        query,
        ivf_center,
        vpqss,
        valids,
        itemss,
        codebooks,
        ivf_roots,
        top_k,
        cluster_idx_dis,
        ordered_vpqss_item_dis,
    )
    print(result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=1024, type=int)
    parser.add_argument("--D", default=128, type=int)
    parser.add_argument("--M", default=8, type=int)
    parser.add_argument("--K", default=16, type=int)
    parser.add_argument("--n_list", default=128, type=int)
    parser.add_argument("--n_probe", default=8, type=int)

    args = parser.parse_args()

    N = args.N
    D = args.D
    M = args.M
    K = args.K
    n_list = args.n_list
    n_probe = args.n_probe
    n = N // n_list
    d = D // M
    bench(D, n_list, M, K, d, n_probe, n)
