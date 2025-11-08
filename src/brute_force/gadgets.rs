use crate::prelude::*;
use crate::utils::nn_gadgets::static_nn_gadget;

pub fn brute_force_gadget(
    builder: &mut CircuitBuilder<F, D>, // builder
    fs_hash: Vec<Target>,               // Fiat-Shamior用的值 (2,)
    src_vecs: Vec<Vec<Target>>,         // (N,D)
    query: Vec<Target>,                 // (D,)
    sorted_idx_dis: Vec<Vec<Target>>,   // (N,2)
) {
    static_nn_gadget(
        builder,
        fs_hash[0],
        fs_hash[1],
        src_vecs,
        query,
        sorted_idx_dis,
    );
}
