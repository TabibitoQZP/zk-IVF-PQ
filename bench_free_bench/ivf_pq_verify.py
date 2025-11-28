"""
函数签名

pub fn ivf_pq_proof(
    ivf_centers: Vec<Vec<u64>>,      // (n_list,D)
    query: Vec<u64>,                 // (D,)
    sorted_idx_dis: Vec<Vec<u64>>,   // (n_list,2)
    filtered_centers: Vec<Vec<u64>>, // (n_probe,D)
    probe_count: Vec<u64>,           // (n_probe,)
    filtered_vecs: Vec<Vec<u64>>,    // (max_sz,M)
    vecs_cluster_hot: Vec<Vec<u64>>, // (max_sz,n_probe)
    codebooks: Vec<Vec<Vec<u64>>>,   // (M,K,d)
) -> Result<(), Box<dyn std::error::Error>> {}
"""

import argparse
import numpy as np
from zk_IVF_PQ.zk_IVF_PQ import py_ivf_pq_verify_proof

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--N", default=1024, type=int)
parser.add_argument("--D", default=128, type=int)
parser.add_argument("--M", default=64, type=int)
parser.add_argument("--K", default=256, type=int)
parser.add_argument("--n_list", default=128, type=int)
parser.add_argument("--n_probe", default=8, type=int)
parser.add_argument("--seed", default=40, type=int)

args = parser.parse_args()

N = args.N
D = args.D
M = args.M
K = args.K
d = D // M
n_list = args.n_list
n_probe = args.n_probe
avg_cnt = N // n_list
max_sz = 2 * avg_cnt * n_probe
seed = args.seed


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
    rng = np.random.default_rng(seed=seed)

    # NOTE: 似乎uint32太小了, 得上int64, 不过不影响证明和测试
    query = rng.integers(0, 15, size=(D,), dtype=np.uint32, endpoint=True)
    ivf_centers = rng.integers(0, 15, size=(n_list, D), dtype=np.uint32, endpoint=True)

    c = np.sum((ivf_centers - query) ** 2, axis=1).astype(np.uint32)
    order = np.argsort(c, kind="stable")
    order = order.astype(np.uint32)

    sorted_idx_dis = np.column_stack((order, c[order]))
    filtered_centers = ivf_centers[order]

    probe_count = np.array([avg_cnt] * n_probe, dtype=np.uint32)
    vecs_cluster_hot = make_block_onehot(max_sz, n_probe, avg_cnt)

    codebooks = np.zeros((M, K, d), dtype=np.uint32)
    filtered_vecs = rng.integers(
        0, K - 1, size=(max_sz, M), dtype=np.uint32, endpoint=True
    )

    result = py_ivf_pq_verify_proof(
        ivf_centers,
        query,
        sorted_idx_dis,
        filtered_centers,
        probe_count,
        filtered_vecs,
        vecs_cluster_hot,
        codebooks,
    )
    print(result)


if __name__ == "__main__":
    print("ivf pq verify")
    bench()
