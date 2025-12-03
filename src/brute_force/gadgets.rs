use crate::hash_gadgets::merkle_tree_gadget;
use crate::prelude::*;
use crate::utils::dis_gadgets::distance;
use crate::utils::nn_gadgets::{comp_gadget, static_nn_gadget};

pub fn rev_gadget(
    builder: &mut CircuitBuilder<F, D>,
    left: Target,
    right: Target,
    rev: Target, // 0代表不换, 1代表换
) -> (Target, Target) {
    let one = builder.one();
    let sub_rev = builder.sub(one, rev);

    let z0 = builder.mul(rev, left);
    let z1 = builder.mul(sub_rev, left);
    let z2 = builder.mul(rev, right);
    let z3 = builder.mul(sub_rev, right);

    let new_left = builder.add(z1, z2);
    let new_right = builder.add(z0, z3);

    (new_left, new_right)
}

pub fn brute_force_gadget(
    builder: &mut CircuitBuilder<F, D>, // builder
    fs_hash: Vec<Target>,               // Fiat-Shamior用的值 (2,)
    src_vecs: Vec<Vec<Target>>,         // (N,D)
    query: Vec<Target>,                 // (D,)
    sorted_idx_dis: Vec<Vec<Target>>,   // (N,2)
) {
    let root = merkle_tree_gadget(builder, src_vecs.clone());
    static_nn_gadget(
        builder,
        fs_hash[0],
        fs_hash[1],
        src_vecs,
        query,
        sorted_idx_dis,
    );
}

pub fn sort_brute_force_gadget(
    builder: &mut CircuitBuilder<F, D>,
    src_vecs: Vec<Vec<Target>>,
    query: Vec<Target>,
    top_k: u64,
) {
    let root = merkle_tree_gadget(builder, src_vecs.clone());
    let N_ = src_vecs.len();
    // 打印
    let mut idxes: Vec<Target> = (0..src_vecs.len())
        .map(|item| builder.constant(F::from_canonical_u64(item as u64)))
        .collect();
    let mut src_dis: Vec<Target> = src_vecs
        .into_iter()
        .map(|raw| distance(builder, raw, query.clone()))
        .collect();
    // 计算
    for i in 0..top_k {
        for j in ((i + 1) as usize..N_).rev() {
            let comp_result = comp_gadget(builder, src_dis[j - 1], src_dis[j]);
            (src_dis[j - 1], src_dis[j]) =
                rev_gadget(builder, src_dis[j - 1], src_dis[j], comp_result);
            // 注意这里也要换
            (idxes[j - 1], idxes[j]) = rev_gadget(builder, idxes[j - 1], idxes[j], comp_result);
        }
    }
}
