"""
函数签名

pub fn sort_brute_force_proof(
    src_vecs: Vec<Vec<u64>>, // (N,D)
    query: Vec<u64>,         // (D,)
    top_k: u64,
) -> Result<(f64, f64, f64, u64, u64, u64), Box<dyn std::error::Error>> {
"""

import argparse
import numpy as np
from zk_IVF_PQ.zk_IVF_PQ import py_sort_brute_force_proof

parser = argparse.ArgumentParser()
parser.add_argument("--N", default=1024, type=int)
parser.add_argument("--D", default=1024, type=int)
parser.add_argument("--k", default=128, type=int)

args = parser.parse_args()

N = args.N
D = args.D
top_k = args.k


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

    query = rng.integers(0, 127, size=(D,), dtype=np.uint32, endpoint=True)
    ivf_centers = rng.integers(0, 127, size=(N, D), dtype=np.uint32, endpoint=True)

    result = py_sort_brute_force_proof(ivf_centers, query, top_k)
    print(result)


if __name__ == "__main__":
    print("sort brute force")
    bench()
