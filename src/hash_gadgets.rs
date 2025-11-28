use crate::prelude::*;
use crate::utils::common_gadgets::static_lookup_gadget;

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

/*
* merkle树回溯, 注意要分方向
* leaf按从左到右的逻辑顺序排, 最左边的方向是全0, 最右边方向是全1
*/
pub fn merkle_back_gadget(
    builder: &mut CircuitBuilder<F, D>,
    leaf: Vec<Target>,      // (n,)
    path: Vec<Vec<Target>>, // (d, 2)
) -> Target {
    let mut curr_target = hash_gadget(builder, leaf);
    let one = builder.one();
    for i in 0..path.len() {
        static_lookup_gadget(builder, path[i][0], vec![0, 1]);
        let b0 = path[i][0]; // b0=0, 则curr_target在左边
        let b1 = builder.sub(one, path[i][0]);

        let v00 = builder.mul(b0, path[i][1]);
        let v01 = builder.mul(b1, path[i][1]);
        let v10 = builder.mul(b0, curr_target);
        let v11 = builder.mul(b1, curr_target);

        // 重构左右
        let left = builder.add(v11, v00);
        let right = builder.add(v01, v10);

        curr_target = hash_gadget(builder, vec![left, right]);
    }
    curr_target
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

pub fn tree_depth(mut leaf_len: usize) -> usize {
    let mut depth: usize = 0;
    while leaf_len > 1 {
        leaf_len /= 2;
        depth += 1;
    }
    depth
}

pub fn hash_tree_gen(mut hash_list: Vec<u64>) -> Vec<u64> {
    let mut hash_len = hash_list.len();
    let mut hash_tree: Vec<u64> = Vec::new();
    hash_tree.extend(hash_list.clone());
    while hash_len > 1 {
        hash_len /= 2;
        let mut curr_hash_list: Vec<u64> = Vec::with_capacity(hash_len);
        for i in 0..hash_len {
            curr_hash_list.push(hash_u64(vec![hash_list[2 * i], hash_list[2 * i + 1]]));
        }
        hash_list = curr_hash_list.clone();
        curr_hash_list.extend(hash_tree);
        hash_tree = curr_hash_list;
    }
    hash_tree
}

pub fn hash_tree_path(mut idx: u64, hash_tree: Vec<u64>) -> Vec<Vec<u64>> {
    let depth = tree_depth((hash_tree.len() + 1) / 2);
    let mut idx_bits: Vec<u64> = Vec::with_capacity(depth);
    for i in 0..depth {
        idx_bits.push(idx - (idx / 2 * 2));
        idx /= 2;
    }
    idx_bits.reverse();
    let mut other_part: Vec<u64> = Vec::with_capacity(depth);
    let mut curr_idx = 0;
    for i in idx_bits.clone() {
        curr_idx = curr_idx * 2 + i + 1;
        if curr_idx % 2 == 0 {
            other_part.push(hash_tree[curr_idx as usize - 1]);
        } else {
            other_part.push(hash_tree[curr_idx as usize + 1]);
        }
    }
    idx_bits.reverse();
    other_part.reverse();
    let mut pairs: Vec<Vec<u64>> = Vec::with_capacity(depth);
    for i in 0..depth {
        pairs.push(vec![idx_bits[i], other_part[i]]);
    }
    pairs
}
