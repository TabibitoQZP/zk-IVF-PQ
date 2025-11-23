use crate::hash_gadgets::{hash_gadget, merkle_tree_gadget};
use crate::prelude::*;

pub fn merkle_commit_gadget(
    builder: &mut CircuitBuilder<F, D>, // builder
    leaves: Vec<Vec<Target>>,           // (2^n,d)
) {
    let root = merkle_tree_gadget(builder, leaves);
    builder.register_public_input(root);
}

pub fn merkle_commit_plain_gadget(
    builder: &mut CircuitBuilder<F, D>, // builder
    leaves: Vec<Vec<Target>>,           // (2^n,d)
) {
    let flat: Vec<Target> = leaves.into_iter().flatten().collect();
    let root = hash_gadget(builder, flat);
    builder.register_public_input(root);
}
