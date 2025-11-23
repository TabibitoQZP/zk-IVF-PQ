use crate::merkle_commit::gadgets::merkle_commit_gadget;
use crate::merkle_commit::gadgets::merkle_commit_plain_gadget;
use crate::prelude::*;

pub fn merkle_commit_proof(leaves: Vec<Vec<u64>>) -> Result<(), Box<dyn std::error::Error>> {
    let pw2 = leaves.len();
    let d = leaves[0].len();

    // 初始化电路
    let mut builder = make_builder();

    // 初始化输入
    let leaves_targets = add_targets_2d(&mut builder, vec![pw2, d]);
    merkle_commit_gadget(&mut builder, leaves_targets.clone());

    // 构建电路
    let mut curr_time = Instant::now();
    let data = builder.build::<C>();
    println!("构建电路耗时: {:?}", curr_time.elapsed());

    // 输入公开输入和witness
    curr_time = Instant::now();
    let mut pw = PartialWitness::new();
    input_targets_2d(&mut pw, leaves_targets, leaves)?;
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

pub fn merkle_commit_plain_proof(leaves: Vec<Vec<u64>>) -> Result<(), Box<dyn std::error::Error>> {
    let pw2 = leaves.len();
    let d = leaves[0].len();

    // 初始化电路
    let mut builder = make_builder();

    // 初始化输入
    let leaves_targets = add_targets_2d(&mut builder, vec![pw2, d]);
    merkle_commit_plain_gadget(&mut builder, leaves_targets.clone());

    // 构建电路
    let mut curr_time = Instant::now();
    let data = builder.build::<C>();
    println!("构建电路耗时: {:?}", curr_time.elapsed());

    // 输入公开输入和witness
    curr_time = Instant::now();
    let mut pw = PartialWitness::new();
    input_targets_2d(&mut pw, leaves_targets, leaves)?;
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
