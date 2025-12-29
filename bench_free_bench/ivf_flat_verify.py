"""
函数签名

pub fn ivf_flat_verify_proof(
    ivf_centers: Vec<Vec<u64>>,    // (n_list,d)
    query: Vec<u64>,               // (d,)
    sorted_idx_dis: Vec<Vec<u64>>, // (n_list,2)
    vecss: Vec<Vec<Vec<u64>>>,     // (n_probe,n,d)
    valids: Vec<Vec<u64>>,         // (n_probe,n)
    itemss: Vec<Vec<u64>>,         // (n_probe,n)
    top_k: usize,                  // 明确取哪top_k
) -> Result<(), Box<dyn std::error::Error>> {}
"""

import argparse

import numpy as np
from zk_IVF_PQ.zk_IVF_PQ import py_ivf_flat_verify_proof


parser = argparse.ArgumentParser()
parser.add_argument("--n_list", default=128, type=int)
parser.add_argument("--n_probe", default=8, type=int)
parser.add_argument("--n", default=64, type=int)
parser.add_argument("--d", default=128, type=int)
parser.add_argument("--top_k", default=10, type=int)
args = parser.parse_args()

n_list = args.n_list
n_probe = args.n_probe
n = args.n
d = args.d
top_k = args.top_k


def bench():
    rng = np.random.default_rng()

    query = rng.integers(0, 127, size=(d,), dtype=np.uint32, endpoint=True)
    ivf_centers = rng.integers(0, 127, size=(n_list, d), dtype=np.uint32, endpoint=True)

    c = np.sum((ivf_centers - query) ** 2, axis=1).astype(np.uint32)
    order = np.argsort(c, kind="stable").astype(np.uint32)
    sorted_idx_dis = np.column_stack((order, c[order]))

    vecss = rng.integers(0, 127, size=(n_probe, n, d), dtype=np.uint32, endpoint=True)
    valids = np.zeros((n_probe, n), dtype=np.uint32)
    for i in range(n_probe):
        cnt = rng.integers(0, n + 1)
        valids[i, :cnt] = 1

    itemss = np.arange(n_probe * n, dtype=np.uint32).reshape(n_probe, n)

    assert 0 <= top_k <= n_probe * n
    result = py_ivf_flat_verify_proof(
        ivf_centers.tolist(),
        query.tolist(),
        sorted_idx_dis.tolist(),
        vecss.tolist(),
        valids.tolist(),
        itemss.tolist(),
        top_k,
    )
    print(result)


if __name__ == "__main__":
    print("ivf flat verify")
    bench()

