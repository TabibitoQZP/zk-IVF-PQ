"""
函数签名

pub fn pq_flat_accel_proof(
    codebooks: Vec<Vec<Vec<u64>>>,       // (M,K,d)
    query: Vec<u64>,                     // (D,)
    pq_vecs: Vec<Vec<u64>>,              // (N,M)
    pq_sub_distances: Vec<Vec<u64>>,     // (N,M)
    unused_table_entries: Vec<Vec<u64>>, // (M*K*N - N*M, 4)
    sorted_idx_dis: Vec<Vec<u64>>,       // (N,2)
) -> Result<(), Box<dyn std::error::Error>> {}
"""

import argparse
import numpy as np

from zk_IVF_PQ.zk_IVF_PQ import py_pq_flat_acc_proof


parser = argparse.ArgumentParser()
parser.add_argument("--N", default=64, type=int)
parser.add_argument("--M", default=32, type=int)
parser.add_argument("--K", default=64, type=int)
parser.add_argument("--D", default=1024, type=int)

args = parser.parse_args()

N = args.N
M = args.M
K = args.K
D = args.D

d = D // M

if N > K:
    raise ValueError("pq_flat_accel circuit requires N <= K so that N*M <= M*K")


def pq_lut(codebooks: np.ndarray, query: np.ndarray) -> np.ndarray:
    M, K, d = codebooks.shape
    q = query.reshape(M, d)
    diff = codebooks - q[:, None, :]
    return np.sum(diff * diff, axis=2)


def pq_sub_distances_from_lut(lut: np.ndarray, codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.intp)
    picked = np.take_along_axis(lut, codes.T, axis=1)
    return picked.T


def pq_distances_from_sub(sub_dists: np.ndarray) -> np.ndarray:
    return sub_dists.sum(axis=1)


def unused_entries(
    lut: np.ndarray,
    pq_vecs: np.ndarray,
) -> list[list[int]]:
    M, K = lut.shape
    N = pq_vecs.shape[0]
    buckets: dict[tuple[int, int], list[int]] = {}
    for vec_idx in range(N):
        for sub_idx in range(M):
            key = (sub_idx, int(pq_vecs[vec_idx, sub_idx]))
            buckets.setdefault(key, []).append(vec_idx)

    rows: list[list[int]] = []
    for sub_idx in range(M):
        for code_idx in range(K):
            dist = int(lut[sub_idx, code_idx])
            used_slots = buckets.get((sub_idx, code_idx))
            used_slots_set = set(used_slots) if used_slots else ()
            for vec_idx in range(N):
                if used_slots and vec_idx in used_slots_set:
                    continue
                rows.append([sub_idx, code_idx, vec_idx, dist])
    return rows


def bench():
    rng = np.random.default_rng()

    codebooks = rng.integers(0, 127, size=(M, K, d), dtype=np.uint32, endpoint=True)
    query = rng.integers(0, 127, size=(D,), dtype=np.uint32, endpoint=True)
    pq_vecs = rng.integers(0, K, size=(N, M), dtype=np.uint32)

    lut = pq_lut(codebooks.astype(np.uint64), query.astype(np.uint64)).astype(np.uint64)
    pq_sub_dists = pq_sub_distances_from_lut(lut, pq_vecs).astype(np.uint64)
    pq_total_dists = pq_distances_from_sub(pq_sub_dists)

    order = np.argsort(pq_total_dists, kind="stable").astype(np.uint32)
    sorted_idx_dis = np.column_stack((order, pq_total_dists[order])).astype(np.uint64)

    unused_rows = unused_entries(lut, pq_vecs)
    expected_unused = M * K * N - N * M
    if len(unused_rows) != expected_unused:
        raise ValueError(
            f"expected {expected_unused} unused rows but built {len(unused_rows)}"
        )
    if unused_rows:
        unused_rows = np.asarray(unused_rows, dtype=np.uint64).tolist()
    else:
        unused_rows = []

    result = py_pq_flat_acc_proof(
        codebooks.astype(np.uint64).tolist(),
        query.astype(np.uint64).tolist(),
        pq_vecs.astype(np.uint64).tolist(),
        pq_sub_dists.astype(np.uint64).tolist(),
        unused_rows,
        sorted_idx_dis.astype(np.uint64).tolist(),
    )
    print(result)


if __name__ == "__main__":
    print("pq flat acc")
    bench()
