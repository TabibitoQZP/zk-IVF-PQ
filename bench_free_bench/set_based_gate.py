"""
函数签名如下
pub fn set_based_gate(
    M: usize,
    K: usize,
    d: usize,
    n_list: usize,
    n_probe: usize,
    n: usize,
    top_k: usize,
    merkled: bool,
) -> usize {}
"""

from zk_IVF_PQ.zk_IVF_PQ import py_set_based_gate
import time

if __name__ == "__main__":
    stime = time.time()
    D = 960
    n_list = 8192
    n_probe = 64
    n = 256
    for B in [8, 16, 32, 64]:
        for log2_k in [1,2,4,8]:
            K = 2**log2_k
            M = B // log2_k
            d = D // M
            gate = py_set_based_gate(M, K, d, n_list, n_probe, n, 64, False)
            print(K,M,gate)
    print(time.time()-stime)
