use crate::hash_gadgets::{hash_gadget, hash_u64, merkle_tree_gadget};
use crate::prelude::*;

pub fn hash_i64(inputs: Vec<i64>) -> u64 {
    let elems: Vec<F> = inputs.into_iter().map(F::from_noncanonical_i64).collect();
    PoseidonHash::hash_no_pad(&elems).elements[0].to_canonical_u64()
}

/*
 * 将cluster内的vpqs打包成merkle承诺
 * 要求vpq个数恰好为n=2^t
 * 顺序(i, j, b_{i,j}, item_{i,j}, v^{PQ}_{i,j})
 */
pub fn merkle_cluster_gadget(
    builder: &mut CircuitBuilder<F, D>, // builder
    cluster_idx: Target,                // cluster对应的索引号
    valid: Vec<Target>,                 // 对应的vpqs是否valid (n,)
    items: Vec<Target>,                 // 需要取出的内容 (n,)
    vpqs: Vec<Vec<Target>>,             // 量化后的向量 (n,M)
) -> Target {
    let leave_len = 4 + vpqs[0].len();
    let n = vpqs.len();

    let mut leaves: Vec<Vec<Target>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut leaf: Vec<Target> = Vec::with_capacity(leave_len);
        let i_target = builder.constant(F::from_canonical_u64(i as u64));
        builder.register_public_input(i_target);
        // 顺序(i, j, b_{i,j}, item_{i,j}, v^{PQ}_{i,j})
        leaf.push(cluster_idx);
        leaf.push(i_target);
        leaf.push(valid[i]);
        leaf.push(items[i]);
        leaf.extend(vpqs[i].clone());
        leaves.push(leaf);
    }
    merkle_tree_gadget(builder, leaves)
}
pub fn merkle_cluster_i64(
    cluster_idx: i64,    // cluster对应的索引号
    valid: Vec<i64>,     // 对应的vpqs是否valid (n,)
    items: Vec<i64>,     // 需要取出的内容 (n,)
    vpqs: Vec<Vec<i64>>, // 量化后的向量 (n,M)
) -> u64 {
    let leave_len = 4 + vpqs[0].len();
    let n = vpqs.len();

    let mut hash_vals: Vec<u64> = Vec::with_capacity(n);
    for i in 0..n {
        let mut leaf: Vec<i64> = Vec::with_capacity(leave_len);
        // 顺序(i, j, b_{i,j}, item_{i,j}, v^{PQ}_{i,j})
        leaf.push(cluster_idx);
        leaf.push(i as i64);
        leaf.push(valid[i]);
        leaf.push(items[i]);
        leaf.extend(vpqs[i].clone());
        hash_vals.push(hash_i64(leaf));
    }

    let mut hash_len = hash_vals.len();
    while hash_len > 1 {
        hash_len /= 2;
        let mut curr_hash_list: Vec<u64> = Vec::with_capacity(hash_len);
        for i in 0..hash_len {
            curr_hash_list.push(hash_u64(vec![hash_vals[2 * i], hash_vals[2 * i + 1]]));
        }
        hash_vals = curr_hash_list;
    }
    hash_vals[0]
}

/*
 * 将cluster索引, merkle根, 中心打包成mekle承诺
 * 顺序 (i, c_i, root_i)
 */
pub fn merkle_ivf_gadget(
    builder: &mut CircuitBuilder<F, D>,
    c: Vec<Vec<Target>>, // cluster的中心 (n_list, D)
    roots: Vec<Target>,  // cluster对应的merkle根 (n_list,)
) -> Target {
    let leave_len = 2 + c[0].len();
    let n_list = c.len();

    let mut leaves: Vec<Vec<Target>> = Vec::with_capacity(n_list);
    for i in 0..n_list {
        let mut leaf: Vec<Target> = Vec::with_capacity(leave_len);
        let i_target = builder.constant(F::from_canonical_u64(i as u64));
        builder.register_public_input(i_target);
        // 顺序(i, c_i, root_i)
        leaf.push(i_target);
        leaf.extend(c[i].clone());
        leaf.push(roots[i]);
        leaves.push(leaf);
    }
    merkle_tree_gadget(builder, leaves)
}

/*
 * codebooks承诺, 只需要哈希一下即可
 */
pub fn commit_codebook_gadget(
    builder: &mut CircuitBuilder<F, D>,
    codebooks: Vec<Vec<Vec<Target>>>, // 码本, 维度为 (M,K,d)
) -> Target {
    let M = codebooks.len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();
    let mut flat_cb: Vec<Target> = Vec::with_capacity(M * K * d);

    for mat in codebooks {
        for row in mat {
            flat_cb.extend(row);
        }
    }
    hash_gadget(builder, flat_cb)
}

pub fn commit_codebook_i64(codebooks: Vec<Vec<Vec<i64>>>) -> u64 {
    let M = codebooks.len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();
    let mut flat_cb: Vec<i64> = Vec::with_capacity(M * K * d);

    for mat in codebooks {
        for row in mat {
            flat_cb.extend(row);
        }
    }
    hash_i64(flat_cb)
}
