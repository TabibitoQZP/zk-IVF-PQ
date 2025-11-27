import argparse
import numpy as np
from zk_IVF_PQ.zk_IVF_PQ import py_circuit_ivf_pq_proof

parser = argparse.ArgumentParser()
parser.add_argument("--N", default=1024, type=int)
parser.add_argument("--D", default=128, type=int)
parser.add_argument("--M", default=64, type=int)
parser.add_argument("--K", default=256, type=int)
parser.add_argument("--n_list", default=128, type=int)
parser.add_argument("--n_probe", default=8, type=int)
parser.add_argument("--top_k", default=32, type=int)
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
max_sz = 2 * avg_cnt  # 注意这里的max_sz是每个cluster的最大数量
seed = args.seed
top_k = args.top_k


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


"""
函数签名
fn py_circuit_ivf_pq_proof(
    query: Vec<i64>,               // 查询向量 (D,)
    ivf_centers: Vec<Vec<i64>>,    // ivf簇中心 *(n_list,D)
    vecs: Vec<Vec<Vec<Vec<i64>>>>, // 这里每个都固定给到 (n_probe,max_sz,M,K)
    hot: Vec<Vec<i64>>,            // 针对vecs是否valid
    codebooks: Vec<Vec<Vec<i64>>>, // 全局码本 (M,K,d)
    top_k: i64,                    // 明确取哪top_k
) -> PyResult<bool>
"""


def bench():
    rng = np.random.default_rng(seed=seed)

    query = rng.integers(0, 127, size=(D,), dtype=np.uint32, endpoint=True)
    ivf_centers = rng.integers(0, 127, size=(n_list, D), dtype=np.uint32, endpoint=True)

    idx = rng.integers(0, K, size=(n_probe, max_sz, M))
    vecs = np.eye(K, dtype=np.uint32)[idx]
    hot = rng.integers(0, 1, size=(n_probe, max_sz), dtype=np.uint32, endpoint=True)
    codebooks = rng.integers(0, 127, size=(M, K, d), dtype=np.uint32, endpoint=True)

    top_k = 32

    result = py_circuit_ivf_pq_proof(query, ivf_centers, vecs, hot, codebooks, top_k)

    print(result)


if __name__ == "__main__":
    print("ivf pq verify")
    bench()
