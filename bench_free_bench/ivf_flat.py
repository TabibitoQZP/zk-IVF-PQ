"""
函数签名

pub fn ivf_flat_proof(
    ivf_centers: Vec<Vec<u64>>,      // (n_list,d)
    query: Vec<u64>,                 // (d,)
    sorted_idx_dis: Vec<Vec<u64>>,   // (n_list,2)
    probe_count: Vec<u64>,           // (n_probe,)
    filtered_vecs: Vec<Vec<u64>>,    // (max_sz,d)
    vecs_cluster_hot: Vec<Vec<u64>>, // (max_sz,n_probe)
) -> Result<(), Box<dyn std::error::Error>> {}
"""

import argparse
import numpy as np
from zk_IVF_PQ.zk_IVF_PQ import py_ivf_flat_proof

parser = argparse.ArgumentParser()
parser.add_argument("--N", default=1024, type=int)
parser.add_argument("--d", default=1024, type=int)
parser.add_argument("--n_list", default=128, type=int)
parser.add_argument("--n_probe", default=8, type=int)

args = parser.parse_args()

N = args.N
d = args.d
n_list = args.n_list
n_probe = args.n_probe
max_sz = 2 * N // n_list * n_probe
avg_cnt = N // n_list


def make_block_onehot(
    max_sz: int, n_probe: int, avg_cnt: int, dtype=np.uint32
) -> np.ndarray:
    out = np.zeros((max_sz, n_probe), dtype=dtype)
    m = min(max_sz, n_probe * avg_cnt)
    if m == 0:
        return out

    rows = np.arange(m)
    cols = rows // avg_cnt
    out[rows, cols] = 1
    return out


def bench():
    rng = np.random.default_rng()

    query = rng.integers(0, 127, size=(d,), dtype=np.uint32, endpoint=True)
    ivf_centers = rng.integers(0, 127, size=(n_list, d), dtype=np.uint32, endpoint=True)

    c = np.sum((ivf_centers - query) ** 2, axis=1).astype(np.uint32)
    order = np.argsort(c, kind="stable")
    order = order.astype(np.uint32)

    sorted_idx_dis = np.column_stack((order, c[order]))

    probe_count = np.array([avg_cnt] * n_probe, dtype=np.uint32)
    vecs_cluster_hot = make_block_onehot(max_sz, n_probe, avg_cnt)

    unorderd_vecs = rng.integers(
        0, 127, size=(max_sz, d), dtype=np.uint32, endpoint=True
    )
    uc = np.sum((unorderd_vecs - query) ** 2, axis=1).astype(np.uint32)
    order = np.argsort(uc, kind="stable")
    filtered_vecs = unorderd_vecs[order]

    result = py_ivf_flat_proof(
        ivf_centers, query, sorted_idx_dis, probe_count, filtered_vecs, vecs_cluster_hot
    )
    print(result)


if __name__ == "__main__":
    bench()
