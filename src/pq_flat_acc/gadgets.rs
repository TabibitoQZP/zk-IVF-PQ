use crate::prelude::*;
use crate::utils::common_gadgets::sum_gadget;
use crate::utils::dis_gadgets::distance;
use crate::utils::set_gadgets::set_equal_gadget;

/// Rebuilds the distance lookup table between the query vector and each codeword.
pub fn codebooks_query_gadget(
    builder: &mut CircuitBuilder<F, D>,
    codebooks: Vec<Vec<Vec<Target>>>,
    query: Vec<Target>,
) -> Vec<Vec<Target>> {
    let m = codebooks.len();
    let k = codebooks[0].len();
    let d = codebooks[0][0].len();

    let mut result: Vec<Vec<Target>> = Vec::with_capacity(m);
    for i in 0..m {
        let mut cur_result: Vec<Target> = Vec::with_capacity(k);
        for j in 0..k {
            let cur_vec = codebooks[i][j].clone();
            let query_slide = query[(i * d)..((i + 1) * d)].to_vec();
            let dis = distance(builder, cur_vec, query_slide);
            cur_result.push(dis);
        }
        result.push(cur_result);
    }
    result
}

fn log2_ceil(x: usize) -> usize {
    if x <= 1 {
        return 0;
    }
    (usize::BITS - (x - 1).leading_zeros()) as usize
}

/// Entry gadget for the accelerated PQ-flat circuit leveraging multiset inclusion.
pub fn pq_flat_acc_gadget(
    builder: &mut CircuitBuilder<F, D>,
    fs_hash: Vec<Target>,
    codebooks: Vec<Vec<Vec<Target>>>,
    query: Vec<Target>,
    pq_vecs: Vec<Vec<Target>>,
    pq_sub_distances: Vec<Vec<Target>>,
    unused_table_entries: Vec<Vec<Target>>,
    sorted_idx_dis: Vec<Vec<Target>>,
) {
    assert_eq!(
        fs_hash.len(),
        4,
        "pq_flat_accel requires four Fiat-Shamir challenges"
    );
    let n = pq_vecs.len();
    assert_eq!(pq_sub_distances.len(), n, "pq distances shape mismatch");

    let m = codebooks.len();
    let k = codebooks[0].len();

    for row in &pq_vecs {
        assert_eq!(row.len(), m, "pq_vec length mismatch");
    }
    for row in &pq_sub_distances {
        assert_eq!(row.len(), m, "pq distance length mismatch");
    }

    let total_table_entries = m * k * n;
    let queried_entries = n * m;
    assert!(
        queried_entries <= total_table_entries,
        "queried entries exceed table capacity"
    );
    assert_eq!(
        unused_table_entries.len(),
        total_table_entries - queried_entries,
        "unexpected number of unused table witnesses"
    );

    let lut = codebooks_query_gadget(builder, codebooks, query);

    // Build the full table entries (subquantizer, code index, slot, distance).
    let mut table_entries: Vec<Vec<Target>> = Vec::with_capacity(total_table_entries);
    for (sub_idx, lut_row) in lut.into_iter().enumerate() {
        let sub_const = builder.constant(F::from_canonical_usize(sub_idx));
        for (code_idx, distance) in lut_row.into_iter().enumerate() {
            let code_const = builder.constant(F::from_canonical_usize(code_idx));
            for slot_idx in 0..n {
                let slot_const = builder.constant(F::from_canonical_usize(slot_idx));
                table_entries.push(vec![sub_const, code_const, slot_const, distance.clone()]);
            }
        }
    }

    // Encode all looked-up entries and sum per PQ vector.
    let mut membership_lhs: Vec<Vec<Target>> = Vec::with_capacity(total_table_entries);
    let mut unsorted_idx_dis: Vec<Vec<Target>> = Vec::with_capacity(n);
    let range_bits = log2_ceil(k);

    for (vec_idx, (codes, dists)) in pq_vecs
        .into_iter()
        .zip(pq_sub_distances.into_iter())
        .enumerate()
    {
        let mut dist_row = Vec::with_capacity(m);
        for (sub_idx, (code, dist)) in codes.into_iter().zip(dists.into_iter()).enumerate() {
            if range_bits > 0 {
                builder.range_check(code, range_bits);
            }
            let sub_const = builder.constant(F::from_canonical_usize(sub_idx));
            let slot_const = builder.constant(F::from_canonical_usize(vec_idx));
            membership_lhs.push(vec![sub_const, code, slot_const, dist.clone()]);
            dist_row.push(dist);
        }

        let total_dis = sum_gadget(builder, dist_row);
        unsorted_idx_dis.push(vec![
            builder.constant(F::from_canonical_usize(vec_idx)),
            total_dis,
        ]);
    }

    for entry in unused_table_entries {
        membership_lhs.push(entry);
    }

    set_equal_gadget(
        builder,
        fs_hash[0],
        fs_hash[1],
        membership_lhs,
        table_entries,
    );

    set_equal_gadget(
        builder,
        fs_hash[2],
        fs_hash[3],
        unsorted_idx_dis,
        sorted_idx_dis.clone(),
    );

    for i in 0..n - 1 {
        let sub_target = builder.sub(sorted_idx_dis[i + 1][1], sorted_idx_dis[i][1]);
        builder.range_check(sub_target, 32);
    }
}
