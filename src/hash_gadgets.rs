use crate::prelude::*;

// 将inputs向量映射为一个u64的哈希值
pub fn hash_u64(inputs: Vec<u64>) -> u64 {
    let elems: Vec<F> = inputs.into_iter().map(F::from_canonical_u64).collect();
    PoseidonHash::hash_no_pad(&elems).elements[0].to_canonical_u64()
}

// 对应前面哈希映射函数的电路
pub fn hash_gadget(builder: &mut CircuitBuilder<F, D>, x: Vec<Target>) -> Target {
    builder.hash_n_to_hash_no_pad::<PoseidonHash>(x).elements[0]
}

// 基于哈希电路构建的merkle tree验证电路
// 这里不会主动加索引, 别搞错了!
pub fn merkle_tree_gadget(builder: &mut CircuitBuilder<F, D>, leaves: Vec<Vec<Target>>) -> Target {
    /*
     * The n is the merkle tree size, assume that it equals 2^t
     * The d is the every element dimension
     */
    let n = leaves.len();
    // let d = leaves[0].len();
    let mut hash_val: Vec<Target> = Vec::with_capacity(n);

    for i in 0..n {
        hash_val.push(hash_gadget(builder, leaves[i].clone()));
    }

    while hash_val.len() > 1 {
        let mut tmp_vec: Vec<Target> = Vec::with_capacity(hash_val.len() / 2);
        for i in 0..hash_val.len() / 2 {
            tmp_vec.push(hash_gadget(
                builder,
                vec![hash_val[2 * i], hash_val[2 * i + 1]],
            ))
        }
        hash_val = tmp_vec;
    }
    hash_val[0]
}

// F-S过程的随机数生成器
pub fn fs_oracle(src: Vec<u64>, n: usize) -> Vec<u64> {
    let mut src_cl: Vec<u64> = Vec::with_capacity(src.len() + n);
    src_cl.extend(src.clone());
    for _ in 0..n {
        src_cl.push(hash_u64(src_cl.clone()));
    }
    src_cl[src.len()..].to_vec()
}
