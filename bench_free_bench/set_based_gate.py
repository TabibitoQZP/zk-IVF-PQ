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

if __name__ == "__main__":
    M=8
    K=16
    d=16
    n_list=256
    n_probe=16
    n=32
    top_k=64
    gate = py_set_based_gate(M,K,d,n_list,n_probe,n,top_k,False)
    print(gate)