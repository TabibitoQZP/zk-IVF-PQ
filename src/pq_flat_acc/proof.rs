use crate::hash_gadgets::fs_oracle;
use crate::pq_flat_acc::gadgets::pq_flat_acc_gadget;
use crate::prelude::*;

pub fn pq_flat_acc_proof(
    codebooks: Vec<Vec<Vec<u64>>>,
    query: Vec<u64>,
    pq_vecs: Vec<Vec<u64>>,
    pq_sub_distances: Vec<Vec<u64>>,
    unused_table_entries: Vec<Vec<u64>>,
    sorted_idx_dis: Vec<Vec<u64>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let D_ = query.len();
    let M = codebooks.len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();
    let N = pq_vecs.len();
    let total_table_entries = M * K * N;
    let queried_entries = N * M;
    let unused_rows = unused_table_entries.len();
    assert_eq!(
        unused_rows,
        total_table_entries - queried_entries,
        "unexpected number of unused table witnesses",
    );

    let fs_hash = fs_oracle(query.clone(), 4);

    let mut builder = make_builder();

    let fs_hash_targets = builder.add_virtual_targets(4);
    let codebooks_targets = add_targets_3d(&mut builder, vec![M, K, d]);
    let query_targets = builder.add_virtual_targets(D_);
    let pq_vecs_targets = add_targets_2d(&mut builder, vec![N, M]);
    let pq_sub_dis_targets = add_targets_2d(&mut builder, vec![N, M]);
    let unused_targets = add_targets_2d(&mut builder, vec![unused_rows, 4]);
    let sorted_idx_dis_targets = add_targets_2d(&mut builder, vec![N, 2]);

    pq_flat_acc_gadget(
        &mut builder,
        fs_hash_targets.clone(),
        codebooks_targets.clone(),
        query_targets.clone(),
        pq_vecs_targets.clone(),
        pq_sub_dis_targets.clone(),
        unused_targets.clone(),
        sorted_idx_dis_targets.clone(),
    );

    public_targets_1d(&mut builder, query_targets.clone());

    let mut curr_time = Instant::now();
    let data = builder.build::<C>();
    println!("构建电路耗时: {:?}", curr_time.elapsed());

    curr_time = Instant::now();
    let mut pw = PartialWitness::new();
    input_targets_1d(&mut pw, fs_hash_targets, fs_hash)?;
    input_targets_3d(&mut pw, codebooks_targets, codebooks)?;
    input_targets_1d(&mut pw, query_targets, query)?;
    input_targets_2d(&mut pw, pq_vecs_targets, pq_vecs)?;
    input_targets_2d(&mut pw, pq_sub_dis_targets, pq_sub_distances)?;
    input_targets_2d(&mut pw, unused_targets, unused_table_entries)?;
    input_targets_2d(&mut pw, sorted_idx_dis_targets, sorted_idx_dis)?;
    println!("输入witness: {:?}", curr_time.elapsed());

    curr_time = Instant::now();
    let proof = data.prove(pw)?;
    println!("证明生成: {:?}", curr_time.elapsed());
    curr_time = Instant::now();
    let _ = data.verify(proof);
    println!("证明验证: {:?}", curr_time.elapsed());
    Ok(())
}

