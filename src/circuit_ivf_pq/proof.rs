use crate::circuit_ivf_pq::gadgets::circuit_ivf_pq_gadget;
use crate::prelude::*;
use crate::utils::metrics::metrics_eval;

pub fn circuit_ivf_pq_proof(
    query: Vec<i64>,                // 查询向量 (D,)
    mut ivf_centers: Vec<Vec<i64>>, // ivf簇中心 *(n_list,D)
    vecs: Vec<Vec<Vec<Vec<i64>>>>,  // 这里每个都固定给到 (n_probe,max_sz,M,K)
    hot: Vec<Vec<i64>>,             // 针对vecs是否valid
    codebooks: Vec<Vec<Vec<i64>>>,  // 全局码本 (M,K,d)
    top_k: i64,                     // 明确取哪top_k
) -> Result<(f64, f64, f64, u64, u64), Box<dyn std::error::Error>> {
    let D_ = query.len();
    let n_list = ivf_centers.len();
    let n_probe = vecs.len();
    let max_sz = vecs[0].len();
    let M = codebooks.len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();

    // 初始化电路
    let mut builder = make_builder();

    // 初始化输入
    let query_targets = builder.add_virtual_targets(D_);
    let ivf_centers_targets = add_targets_2d(&mut builder, vec![n_list, D_]);
    let vecs_targets = add_targets_4d(&mut builder, vec![n_probe, max_sz, M, K]);
    let hot_targets = add_targets_2d(&mut builder, vec![n_probe, max_sz]);
    let codebooks_targets = add_targets_3d(&mut builder, vec![M, K, d]);

    // 构建电路
    circuit_ivf_pq_gadget(
        &mut builder,
        query_targets.clone(),
        ivf_centers_targets.clone(),
        vecs_targets.clone(),
        hot_targets.clone(),
        codebooks_targets.clone(),
        top_k,
    );

    // 设置公开输入和witness
    public_targets_1d(&mut builder, query_targets.clone());

    // 输入公开输入和witness
    let curr_time = Instant::now();
    let mut pw = PartialWitness::new();
    input_targets_1d_sign(&mut pw, query_targets, query)?;
    input_targets_2d_sign(&mut pw, ivf_centers_targets, ivf_centers)?;
    input_targets_4d_sign(&mut pw, vecs_targets, vecs)?;
    input_targets_2d_sign(&mut pw, hot_targets, hot)?;
    input_targets_3d_sign(&mut pw, codebooks_targets, codebooks)?;
    println!("输入witness: {:?}", curr_time.elapsed());

    // 整体测试
    let (build_time, prove_time, verify_time, proof_size, memory_used) = metrics_eval(builder, pw)?;

    Ok((build_time, prove_time, verify_time, proof_size, memory_used))
}
