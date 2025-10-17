use crate::prelude::*;

pub fn hash_u64(inputs: Vec<u64>) -> u64 {
    let elems: Vec<F> = inputs.into_iter().map(F::from_canonical_u64).collect();
    PoseidonHash::hash_no_pad(&elems).elements[0].to_canonical_u64()
}

pub fn hash_gadget(builder: &mut CircuitBuilder<F, D>, x: Vec<Target>) -> Target {
    builder.hash_n_to_hash_no_pad::<PoseidonHash>(x).elements[0]
}

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

pub fn set_equal_gadget(
    builder: &mut CircuitBuilder<F, D>,
    r: Target,
    t: Target,
    src: Vec<Vec<Target>>,
    dst: Vec<Vec<Target>>,
) {
    /*
     * r: F-S challenge
     * t: F-S hash value, used to prove permutation
     * n: set size
     * d: set element dimension
     */
    let n = src.len();
    let d = src[0].len();

    let mut fac_src: Vec<Target> = Vec::with_capacity(n);
    let mut fac_dst: Vec<Target> = Vec::with_capacity(n);
    for i in 0..n {
        let mut cur_targets = src[i][0];
        for j in 1..d {
            cur_targets = builder.mul(t, cur_targets);
            cur_targets = builder.add(cur_targets, src[i][j]);
        }
        fac_src.push(builder.sub(r, cur_targets));

        let mut cur_targetd = dst[i][0];
        for j in 1..d {
            cur_targetd = builder.mul(t, cur_targetd);
            cur_targetd = builder.add(cur_targetd, dst[i][j]);
        }
        fac_dst.push(builder.sub(r, cur_targetd));
    }

    let prod_src = builder.mul_many(fac_src);
    let prod_dst = builder.mul_many(fac_dst);

    builder.connect(prod_src, prod_dst);
}
