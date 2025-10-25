use crate::hash_gadgets::fs_oracle;
use crate::ivf_pq::gadgets::ivf_pq_gadget;
use crate::prelude::*;

pub fn ivf_pq_proof(
    ivf_centers: Vec<Vec<u64>>,      // (n_list,D)
    query: Vec<u64>,                 // (D,)
    sorted_idx_dis: Vec<Vec<u64>>,   // (n_list,2)
    filtered_centers: Vec<Vec<u64>>, // (n_probe,D)
    probe_count: Vec<u64>,           // (n_probe,)
    filtered_vecs: Vec<Vec<u64>>,    // (max_sz,M)
    vecs_cluster_hot: Vec<Vec<u64>>, // (max_sz,n_probe)
    codebooks: Vec<Vec<Vec<u64>>>,   // (M,K,d)
) -> Result<(), Box<dyn std::error::Error>> {
    // 初始化维度信息
    let n_list = ivf_centers.len();
    let D_ = query.len();
    let n_probe = probe_count.len();
    let max_sz = filtered_vecs.len();
    let M = filtered_vecs[0].len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();

    // 初始化F-S哈希值
    let fs_hash = fs_oracle(query.clone(), 2);

    // 初始化电路
    let mut builder = make_builder();

    // 初始化输入
    let fs_hash_targets = builder.add_virtual_targets(2);
    let ivf_centers_targets = add_targets_2d(&mut builder, vec![n_list, D_]);
    let query_targets = builder.add_virtual_targets(D_);
    let sorted_idx_dis_targets = add_targets_2d(&mut builder, vec![n_list, 2]);
    let filtered_centers_targets = add_targets_2d(&mut builder, vec![n_probe, D_]);
    let probe_count_targets = builder.add_virtual_targets(n_probe);
    let filtered_vecs_targets = add_targets_2d(&mut builder, vec![max_sz, M]);
    let vecs_cluster_hot_targets = add_targets_2d(&mut builder, vec![max_sz, n_probe]);
    let codebooks_targets = add_targets_3d(&mut builder, vec![M, K, d]);

    // 构建电路
    ivf_pq_gadget(
        &mut builder,
        fs_hash_targets.clone(),
        ivf_centers_targets.clone(),
        query_targets.clone(),
        sorted_idx_dis_targets.clone(),
        filtered_centers_targets.clone(),
        probe_count_targets.clone(),
        filtered_vecs_targets.clone(),
        vecs_cluster_hot_targets.clone(),
        codebooks_targets.clone(),
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
    input_targets_2d(&mut pw, ivf_centers_targets, ivf_centers)?;
    input_targets_1d(&mut pw, query_targets, query)?;
    input_targets_2d(&mut pw, sorted_idx_dis_targets, sorted_idx_dis)?;
    input_targets_2d(&mut pw, filtered_centers_targets, filtered_centers)?;
    input_targets_1d(&mut pw, probe_count_targets, probe_count)?;
    input_targets_2d(&mut pw, filtered_vecs_targets, filtered_vecs)?;
    input_targets_2d(&mut pw, vecs_cluster_hot_targets, vecs_cluster_hot)?;
    input_targets_3d(&mut pw, codebooks_targets, codebooks)?;
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
