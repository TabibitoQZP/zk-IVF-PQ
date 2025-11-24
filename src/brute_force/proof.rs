use crate::brute_force::gadgets::{brute_force_gadget, sort_brute_force_gadget};
use crate::hash_gadgets::fs_oracle;
use crate::prelude::*;

pub fn brute_force_proof(
    src_vecs: Vec<Vec<u64>>,       // (N,D)
    query: Vec<u64>,               // (D,)
    sorted_idx_dis: Vec<Vec<u64>>, // (N,2)
) -> Result<(), Box<dyn std::error::Error>> {
    // 初始化维度信息
    let N = src_vecs.len();
    let D_ = src_vecs[0].len();

    // 初始化F-S哈希值
    let fs_hash = fs_oracle(query.clone(), 2);

    // 初始化电路
    let mut builder = make_builder();

    // 初始化输入
    let fs_hash_targets = builder.add_virtual_targets(2);
    let src_vecs_targets = add_targets_2d(&mut builder, vec![N, D_]);
    let query_targets = builder.add_virtual_targets(D_);
    let sorted_idx_dis_targets = add_targets_2d(&mut builder, vec![N, 2]);

    // 构建电路
    brute_force_gadget(
        &mut builder,
        fs_hash_targets.clone(),
        src_vecs_targets.clone(),
        query_targets.clone(),
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
    input_targets_2d(&mut pw, src_vecs_targets, src_vecs)?;
    input_targets_1d(&mut pw, query_targets, query)?;
    input_targets_2d(&mut pw, sorted_idx_dis_targets, sorted_idx_dis)?;
    println!("输入witness: {:?}", curr_time.elapsed());

    // 证明生成和验证
    curr_time = Instant::now();
    let proof = data.prove(pw)?;
    println!("证明生成: {:?}", curr_time.elapsed());

    // 证明大小
    let compressed_proof = data.compress(proof.clone())?;
    let compressed_bytes = compressed_proof.to_bytes();
    println!("证明大小: {}B", compressed_bytes.len());

    curr_time = Instant::now();
    let _ = data.verify(proof);
    println!("证明验证: {:?}", curr_time.elapsed());
    Ok(())
}

pub fn sort_brute_force_proof(
    src_vecs: Vec<Vec<u64>>, // (N,D)
    query: Vec<u64>,         // (D,)
    top_k: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    // 初始化维度信息
    let N = src_vecs.len();
    let D_ = src_vecs[0].len();

    // 初始化F-S哈希值
    let fs_hash = fs_oracle(query.clone(), 2);

    // 初始化电路
    let mut builder = make_builder();

    // 初始化输入
    let src_vecs_targets = add_targets_2d(&mut builder, vec![N, D_]);
    let query_targets = builder.add_virtual_targets(D_);

    // 构建电路
    sort_brute_force_gadget(
        &mut builder,
        src_vecs_targets.clone(),
        query_targets.clone(),
        top_k,
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
    input_targets_2d(&mut pw, src_vecs_targets, src_vecs)?;
    input_targets_1d(&mut pw, query_targets, query)?;
    println!("输入witness: {:?}", curr_time.elapsed());

    // 证明生成和验证
    curr_time = Instant::now();
    let proof = data.prove(pw)?;
    println!("证明生成: {:?}", curr_time.elapsed());

    // 证明大小
    let compressed_proof = data.compress(proof.clone())?;
    let compressed_bytes = compressed_proof.to_bytes();
    println!("证明大小: {}B", compressed_bytes.len());

    curr_time = Instant::now();
    let _ = data.verify(proof);
    println!("证明验证: {:?}", curr_time.elapsed());
    Ok(())
}
