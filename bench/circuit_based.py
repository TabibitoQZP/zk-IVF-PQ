import numpy as np
from zk_IVF_PQ.zk_IVF_PQ import py_circuit_based_with_merkle

from bench import data_gen


"""
函数签名

pub fn circuit_based_ivf_pq_proof(
    query: Vec<i64>,               // 查询向量 (D,)
    mut ivf_center: Vec<Vec<i64>>, // ivf簇中心 (n_list,D)
    cluster_idxes: Vec<i64>,       // 簇索引 (n_probe,)
    vpqss: Vec<Vec<Vec<i64>>>,     // 这里给原始向量, 手动改one-hot (n_probe,n,M)
    valids: Vec<Vec<i64>>,         // vpqss中向量是否valid (n_probe,n)
    itemss: Vec<Vec<i64>>,         // vpqss中向量对应的查询量 (n_probe,n)
    codebooks: Vec<Vec<Vec<i64>>>, // 全局码本 (M,K,d)
    ivf_roots: Vec<u64>,           // 这里给一下ivf各个root, 用来手算和还原数据 (n_list,)
    top_k: i64,                    // 明确取哪top_k
) -> Result<(f64, f64, f64, u64, u64), Box<dyn std::error::Error>>
"""


def bench(D, n_list, M, K, d, n_probe, n, top_k=64):
    """
    这里的bench本质上要求zk系统自主完成一整套ivf-pq的计算,
    所以不需要实际运算, 给出结果即可
    """
    (
        query,
        ivf_center,
        cluster_idxes,
        vpqss,
        valids,
        itemss,
        codebooks,
        ivf_roots,
    ) = data_gen(D, n_list, M, K, d, n_probe, n)

    result = py_circuit_based_with_merkle(
        query,
        ivf_center,
        cluster_idxes,
        vpqss,
        valids,
        itemss,
        codebooks,
        ivf_roots,
        top_k,
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
