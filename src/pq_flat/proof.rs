use crate::hash_gadgets::fs_oracle;
use crate::pq_flat::gadgets::pq_flat_gadget;
use crate::prelude::*;

pub fn pq_flat_proof(
    codebooks: Vec<Vec<Vec<u64>>>, // (M,K,d)
    query: Vec<u64>,               // (D,)
    pq_vecs: Vec<Vec<u64>>,        // (N,M)
    sorted_idx_dis: Vec<Vec<u64>>, // (N,2)
) -> Result<(), Box<dyn std::error::Error>> {
    let D_ = query.len();
    let M = codebooks.len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();
    let N = pq_vecs.len();

    // 初始化F-S哈希值
    let fs_hash = fs_oracle(query.clone(), 2);

    // 初始化电路
    let mut builder = make_builder();

    // 初始化输入
    let fs_hash_targets = builder.add_virtual_targets(2);
    let codebooks_targets = add_targets_3d(&mut builder, vec![M, K, d]);
    let query_targets = builder.add_virtual_targets(D_);
    let pq_vecs_targets = add_targets_2d(&mut builder, vec![N, M]);
    let sorted_idx_dis_targets = add_targets_2d(&mut builder, vec![N, 2]);

    // 构建电路
    pq_flat_gadget(
        &mut builder,
        fs_hash_targets.clone(),
        codebooks_targets.clone(),
        query_targets.clone(),
        pq_vecs_targets.clone(),
        sorted_idx_dis_targets.clone(),
    );

    // 设置公开输入和witness
    public_targets_1d(&mut builder, query_targets.clone());

    // 构建电路
    let mut curr_time = Instant::now();
    let data = builder.build::<C>();
    println!("构建电路耗时: {:?}", curr_time.elapsed());

    // 输入公开输入和witness
    curr_time = Instant::now();
    let mut pw = PartialWitness::new();
    input_targets_1d(&mut pw, fs_hash_targets, fs_hash)?;
    input_targets_3d(&mut pw, codebooks_targets, codebooks)?;
    input_targets_1d(&mut pw, query_targets, query)?;
    input_targets_2d(&mut pw, pq_vecs_targets, pq_vecs)?;
    input_targets_2d(&mut pw, sorted_idx_dis_targets, sorted_idx_dis)?;
    println!("输入witness: {:?}", curr_time.elapsed());

    // 证明生成和验证
    curr_time = Instant::now();
    let proof = data.prove(pw)?;
    println!("证明生成: {:?}", curr_time.elapsed());
    curr_time = Instant::now();
    let _ = data.verify(proof);
    println!("证明验证: {:?}", curr_time.elapsed());
    Ok(())
}
