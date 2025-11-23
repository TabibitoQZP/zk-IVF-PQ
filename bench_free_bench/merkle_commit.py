import argparse
import numpy as np
from zk_IVF_PQ.zk_IVF_PQ import py_merkle_commit_proof

parser = argparse.ArgumentParser()
parser.add_argument("--N", default=1024 * 1024, type=int)
parser.add_argument("--D", default=1024, type=int)

args = parser.parse_args()

N = args.N
D = args.D


def bench():
    rng = np.random.default_rng()

    leaves = rng.integers(0, 127, size=(N, D), dtype=np.uint32, endpoint=True)
    result = py_merkle_commit_proof(leaves)

    print(result)


if __name__ == "__main__":
    bench()
