use crate::prelude::*;
use crate::utils::dis_gadgets::distance;
use crate::utils::set_gadgets::set_equal_gadget;

/// Implements a binary-tree random access optimized for power-of-two tables.
///
/// Plonky2 的 `builder.random_access` 会在一个步骤里复制所有候选项，
/// 会消耗大量的门数量。这里我们使用底层的布线接口，通过逐层二分的
/// 方式仅复制两两成对的候选，从而显式控制需要的 `select` 次数。
fn pow2_random_access(
    builder: &mut CircuitBuilder<F, D>,
    index: Target,
    table: Vec<Target>,
) -> Target {
    let len = table.len();
    assert!(len.is_power_of_two(), "table length must be a power of two");

    if len == 1 {
        return table.into_iter().next().expect("non-empty table");
    }

    let bit_width = len.trailing_zeros() as usize;
    builder.range_check(index, bit_width);

    let bits = builder.split_le(index, bit_width);
    let mut layer = table;

    for bit in bits {
        let mut next = Vec::with_capacity(layer.len() / 2);
        for pair in layer.chunks(2) {
            let left = pair[0];
            let right = pair[1];
            let selected = builder.select(bit, right, left);
            next.push(selected);
        }
        layer = next;
    }

    debug_assert_eq!(layer.len(), 1);
    layer.into_iter().next().expect("selection result")
}

/// Rebuilds the distance lookup table between the query vector and each codeword.
pub fn codebooks_query_gadget(
    builder: &mut CircuitBuilder<F, D>,
    codebooks: Vec<Vec<Vec<Target>>>,
    query: Vec<Target>,
) -> Vec<Vec<Target>> {
    let M = codebooks.len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();

    let mut result: Vec<Vec<Target>> = Vec::with_capacity(M);
    for i in 0..M {
        let mut cur_result: Vec<Target> = Vec::with_capacity(K);
        for j in 0..K {
            let cur_vec = codebooks[i][j].clone();
            let query_slide = query[(i * d)..((i + 1) * d)].to_vec();
            let dis = distance(builder, cur_vec, query_slide);
            cur_result.push(dis);
        }
        result.push(cur_result);
    }
    result
}

/// Uses the binary-tree random access to accumulate distances for a PQ code.
pub fn accelerated_lut_code_gadget(
    builder: &mut CircuitBuilder<F, D>,
    lut: &[Vec<Target>],
    code: Vec<Target>,
) -> Target {
    let mut total_dis = builder.zero();

    for (row, idx) in lut.iter().zip(code.into_iter()) {
        let dis = pow2_random_access(builder, idx, row.clone());
        total_dis = builder.add(total_dis, dis);
    }

    total_dis
}

/// Entry gadget for the accelerated PQ-flat circuit.
pub fn pq_flat_accel_gadget(
    builder: &mut CircuitBuilder<F, D>,
    fs_hash: Vec<Target>,
    codebooks: Vec<Vec<Vec<Target>>>,
    query: Vec<Target>,
    pq_vecs: Vec<Vec<Target>>,
    sorted_idx_dis: Vec<Vec<Target>>,
) {
    let N = pq_vecs.len();

    let lut = codebooks_query_gadget(builder, codebooks, query);

    let mut unsorted_idx_dis: Vec<Vec<Target>> = Vec::with_capacity(N);
    for (i, pq_vec) in pq_vecs.into_iter().enumerate() {
        let total_dis = accelerated_lut_code_gadget(builder, &lut, pq_vec);
        unsorted_idx_dis.push(vec![
            builder.constant(F::from_canonical_u64(i as u64)),
            total_dis,
        ]);
    }

    set_equal_gadget(
        builder,
        fs_hash[0],
        fs_hash[1],
        unsorted_idx_dis,
        sorted_idx_dis.clone(),
    );

    for i in 0..N - 1 {
        let sub_target = builder.sub(sorted_idx_dis[i + 1][1], sorted_idx_dis[i][1]);
        builder.range_check(sub_target, 32);
    }
}
