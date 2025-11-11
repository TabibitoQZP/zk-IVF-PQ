"""
函数签名

pub fn pq_flat_proof(
    codebooks: Vec<Vec<Vec<u64>>>, // (M,K,d)
    query: Vec<u64>,               // (D,)
    pq_vecs: Vec<Vec<u64>>,        // (N,M)
    sorted_idx_dis: Vec<Vec<u64>>, // (N,2)
) -> Result<(), Box<dyn std::error::Error>> {}
"""

import argparse
import numpy as np
from zk_IVF_PQ.zk_IVF_PQ import py_pq_flat_com_proof

parser = argparse.ArgumentParser()
parser.add_argument("--N", default=1024, type=int)
parser.add_argument("--M", default=32, type=int)
parser.add_argument("--K", default=256, type=int)  # 受限于plonky2设计, 最多只能64
parser.add_argument("--D", default=1024, type=int)

args = parser.parse_args()

N = args.N
M = args.M
K = args.K
D = args.D
d = D // M


def pq_lut(codebooks: np.ndarray, query: np.ndarray) -> np.ndarray:
    M, K, d = codebooks.shape
    q = query.reshape(M, d)  # (M, d)
    diff = codebooks - q[:, None, :]  # (M, K, d)
    return np.sum(diff * diff, axis=2)  # (M, K)


def pq_distances_from_lut(lut: np.ndarray, codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.intp)
    picked = np.take_along_axis(lut, codes.T, axis=1)  # (M, N)
    return picked.sum(axis=0)  # (N,)


def bench():
    rng = np.random.default_rng()

    codebooks = rng.integers(0, 127, size=(M, K, d), dtype=np.uint32, endpoint=True)
    query = rng.integers(0, 127, size=(D,), dtype=np.uint32, endpoint=True)
    pq_vecs = rng.integers(0, 16, size=(N, M), dtype=np.uint32, endpoint=True)

    lut = pq_lut(codebooks, query).astype(np.uint32)
    dis = pq_distances_from_lut(lut, pq_vecs).astype(np.uint32)
    order = np.argsort(dis, kind="stable")
    order = order.astype(np.uint32)

    sorted_idx_dis = np.column_stack((order, dis[order]))

    result = py_pq_flat_com_proof(codebooks, query, pq_vecs, sorted_idx_dis)
    print(result)


if __name__ == "__main__":
    print("pq flat com")
    bench()
