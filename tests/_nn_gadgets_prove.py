import argparse
import numpy as np
from zk_IVF_PQ.zk_IVF_PQ import py_nn_prove, single_hash

parser = argparse.ArgumentParser()
parser.add_argument("--N", default=128, type=int)
parser.add_argument("--D", default=128, type=int)

args = parser.parse_args()
N = args.N
D = args.D


def merkle_root(src_vecs):
    idx = np.arange(N, dtype=np.uint64).reshape(N, 1)
    src_idx_vecs = np.hstack((idx, src_vecs))

    hash_vals = []
    for i in range(src_idx_vecs.shape[0]):
        hash_vals.append(single_hash(src_idx_vecs[i]))
    while len(hash_vals) > 1:
        sz = len(hash_vals) // 2
        tmp = []
        for i in range(sz):
            tmp.append(single_hash([hash_vals[2 * i], hash_vals[2 * i + 1]]))
        hash_vals = tmp
    return hash_vals[0]


if __name__ == "__main__":
    print(f"N: {N}, D: {D}")
    rng = np.random.default_rng(42)  # 可选：设随机种子
    src_vecs = rng.integers(low=0, high=128, size=(N, D), dtype=np.uint64)
    query = rng.integers(low=0, high=128, size=(D,), dtype=np.uint64)
    c = np.sum((src_vecs - query) ** 2, axis=1).astype(np.uint64)
    # print(c.dtype)
    order = np.argsort(c, kind="stable")
    order = order.astype(np.uint64)
    # print(order.dtype)
    sorted_idx_dis = np.column_stack((order, c[order]))

    root = merkle_root(src_vecs)

    # print(sorted_idx_dis.dtype)
    res = py_nn_prove(src_vecs, query, root, sorted_idx_dis)
    print(res)
