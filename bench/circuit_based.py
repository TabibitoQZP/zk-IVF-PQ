import argparse
import numpy as np
from zk_IVF_PQ.zk_IVF_PQ import py_circuit_based_with_merkle, single_hash

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


MAX_VAL = 65535


def cluster_gen(idx, n, M, K, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    vpqs = rng.integers(0, K, size=(n, M), dtype=np.int64, endpoint=False)
    valid = rng.integers(0, 1, size=(n), dtype=np.int64, endpoint=True)
    items = rng.integers(0, MAX_VAL, size=(n), dtype=np.int64, endpoint=True)

    hash_list = []
    for i in range(n):
        left = np.array([idx, i, valid[i], items[i]], dtype=np.int64)
        hash_list.append(single_hash(np.concatenate([left, vpqs[i]])))

    while len(hash_list) > 1:
        hash_list = [
            single_hash([hash_list[2 * i], hash_list[2 * i + 1]])
            for i in range(len(hash_list) // 2)
        ]
    return vpqs, valid, items, hash_list[0]


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


def bench():
    rng = np.random.default_rng()

    query = rng.integers(0, MAX_VAL, size=(D,), dtype=np.int64, endpoint=True)
    ivf_center = rng.integers(
        0, MAX_VAL, size=(n_list, D), dtype=np.int64, endpoint=True
    )
    codebooks = rng.integers(0, MAX_VAL, size=(M, K, d), dtype=np.int64, endpoint=True)

    c = np.sum((ivf_center - query) ** 2, axis=1).astype(np.int64)
    order = np.argsort(c, kind="stable")
    order = order.astype(np.int64)
    cluster_idxes = order[:n_probe]

    vpqss = []
    valids = []
    itemss = []
    ivf_roots = []
    for i in range(n_list):
        vpqs, valid, items, curr_root = cluster_gen(i, n, M, K, rng)
        vpqss.append(vpqs)
        valids.append(valid)
        itemss.append(items)
        ivf_roots.append(curr_root)

    ivf_roots = np.array(ivf_roots, dtype=np.uint64)
    vpqss = np.stack([vpqss[i] for i in cluster_idxes], axis=0)
    valids = np.stack([valids[i] for i in cluster_idxes], axis=0)
    itemss = np.stack([itemss[i] for i in cluster_idxes], axis=0)

    result = py_circuit_based_with_merkle(
        query,
        ivf_center,
        cluster_idxes,
        vpqss,
        valids,
        itemss,
        codebooks,
        ivf_roots,
        64,
    )
    print(result)
    return result


if __name__ == "__main__":
    bench()
