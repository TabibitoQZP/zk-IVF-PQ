use crate::hash_gadgets::merkle_back_gadget;
use crate::merkle_ver::ivf_pq_merkle::{
    commit_codebook_gadget, merkle_cluster_gadget, merkle_ivf_gadget,
};
use crate::prelude::*;

pub fn standalone_commitment_gadget(
    builder: &mut CircuitBuilder<F, D>,
    query: Vec<Target>,                   // 查询向量 (D,)
    root: Target,                         // 总的根
    codebooks_root: Target,               // codebooks对应的根
    codebooks: Vec<Vec<Vec<Target>>>,     // codebooks (M,K,d)
    mut ivf_center: Vec<Vec<Target>>,     // cluster的中心, 注意排序过程需要mut (n_list, D)
    ivf_roots: Vec<Target>,               // cluster对应的merkle根 (n_list,)
    cluster_idxes: Vec<Target>,           // cluster对应的索引号 (n_probe,)
    cluster_center: Vec<Vec<Target>>,     // cluster中心也要提供 (n_probe,D)
    valids: Vec<Vec<Target>>,             // 对应的vpqs是否valid (n_probe,n)
    itemss: Vec<Vec<Target>>,             // 需要取出的内容 (n_probe,n)
    cluster_pairs: Vec<Vec<Vec<Target>>>, // merkle树的路径
    vpqss: Vec<Vec<Vec<Target>>>,         // 量化后的向量 (n_probe,n,M)
) {
    // 基本变量
    let D_ = query.len();
    let M = codebooks.len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();
    let n_list = ivf_center.len();
    let n_probe = cluster_idxes.len();
    let n = valids[0].len();

    // # 承诺部分
    // 0. 正式验证之前, 将root和codebooks_root公开
    builder.register_public_input(root);
    builder.register_public_input(codebooks_root);
    // 1. 验证ivf中心承诺
    let root_target = merkle_ivf_gadget(builder, ivf_center.clone(), ivf_roots);
    builder.connect(root_target, root);
    // 2. 验证codebooks承诺
    let codebooks_root_target = commit_codebook_gadget(builder, codebooks.clone());
    builder.connect(codebooks_root_target, codebooks_root);
    // 3. 计算每个给定的merkle根承诺
    let mut ivf_root_targets: Vec<Target> = Vec::with_capacity(n_probe);
    for i in 0..n_probe {
        let curr_root = merkle_cluster_gadget(
            builder,
            cluster_idxes[i].clone(),
            valids[i].clone(),
            itemss[i].clone(),
            vpqss[i].clone(),
        );
        ivf_root_targets.push(curr_root);
    }
    // 验证 cluster_idxes, cluster_center, ivf_root_targets构成的集合满足merkle路径
    for i in 0..n_probe {
        let mut curr_leaf: Vec<Target> = Vec::with_capacity(D_ + 2);
        curr_leaf.push(cluster_idxes[i].clone());
        curr_leaf.extend(cluster_center[i].clone());
        curr_leaf.push(ivf_root_targets[i].clone());
        merkle_back_gadget(builder, curr_leaf, cluster_pairs[i].clone());
    }
}

use crate::hash_gadgets::{hash_tree_gen, hash_tree_path, tree_depth};
use crate::merkle_ver::circuit_based_proof::hash_ivf_center;
use crate::merkle_ver::ivf_pq_merkle::commit_codebook_i64;
use crate::utils::metrics::metrics_eval;

pub fn standalone_commitment_proof(
    query: Vec<i64>,               // 查询向量 (D,)
    mut ivf_center: Vec<Vec<i64>>, // ivf簇中心 (n_list,D)
    cluster_idxes: Vec<i64>,       // 簇索引 (n_probe,)
    vpqss: Vec<Vec<Vec<i64>>>,     // 这里给原始向量, 手动改one-hot (n_probe,n,M)
    valids: Vec<Vec<i64>>,         // vpqss中向量是否valid (n_probe,n)
    itemss: Vec<Vec<i64>>,         // vpqss中向量对应的查询量 (n_probe,n)
    codebooks: Vec<Vec<Vec<i64>>>, // 全局码本 (M,K,d)
    ivf_roots: Vec<u64>,           // 这里给一下ivf各个root, 用来手算和还原数据 (n_list,)
) -> Result<(f64, f64, f64, u64, u64), Box<dyn std::error::Error>> {
    let D_ = query.len();
    let M = codebooks.len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();
    let n_list = ivf_center.len();
    let n_probe = vpqss.len();
    let n = vpqss[0].len();

    let codebooks_root = commit_codebook_i64(codebooks.clone());
    let cluster_center: Vec<Vec<i64>> = cluster_idxes
        .clone()
        .into_iter()
        .map(|item| ivf_center[item as usize].clone())
        .collect();

    // 计算ivf的hash树
    let hash_list: Vec<u64> = (0..n_list)
        .map(|i| hash_ivf_center(i, ivf_center[i].clone(), ivf_roots[i]))
        .collect();
    let depth = tree_depth(hash_list.len());
    let hash_tree = hash_tree_gen(hash_list);
    let root = hash_tree[0]; // 0号是全局root

    // 计算cluster的root
    let cluster_roots: Vec<u64> = (0..n_probe)
        .map(|item| ivf_roots[cluster_idxes[item] as usize])
        .collect();
    // 计算root对应的hash路径
    let mut cluster_pairs: Vec<Vec<Vec<u64>>> = Vec::with_capacity(n_probe);
    for i in 0..n_probe {
        let pairs = hash_tree_path(cluster_idxes[i] as u64, hash_tree.clone());
        cluster_pairs.push(pairs);
    }

    let mut builder = make_builder();
    let query_targets = builder.add_virtual_targets(D_);
    let root_targets = builder.add_virtual_target();
    let codebooks_root_targets = builder.add_virtual_target();
    let codebooks_targets = add_targets_3d(&mut builder, vec![M, K, d]);
    let ivf_center_targets = add_targets_2d(&mut builder, vec![n_list, D_]);
    let ivf_roots_targets = builder.add_virtual_targets(n_list);
    let cluster_idxes_targets = builder.add_virtual_targets(n_probe);
    let cluster_center_targets = add_targets_2d(&mut builder, vec![n_probe, D_]);
    let valids_targets = add_targets_2d(&mut builder, vec![n_probe, n]);
    let itemss_targets = add_targets_2d(&mut builder, vec![n_probe, n]);
    let vpqss_targets = add_targets_3d(&mut builder, vec![n_probe, n, M]);
    let cluster_pairs_targets = add_targets_3d(&mut builder, vec![n_probe, depth, 2]);

    standalone_commitment_gadget(
        &mut builder,
        query_targets.clone(),
        root_targets.clone(),
        codebooks_root_targets.clone(),
        codebooks_targets.clone(),
        ivf_center_targets.clone(),
        ivf_roots_targets.clone(),
        cluster_idxes_targets.clone(),
        cluster_center_targets.clone(),
        valids_targets.clone(),
        itemss_targets.clone(),
        cluster_pairs_targets.clone(),
        vpqss_targets.clone(),
    );

    public_targets_1d(&mut builder, query_targets.clone());

    let curr_time = Instant::now();
    let mut pw = PartialWitness::new();
    input_targets_1d_sign(&mut pw, query_targets, query)?;
    input_targets_0d(&mut pw, root_targets, root)?;
    input_targets_0d(&mut pw, codebooks_root_targets, codebooks_root)?;
    input_targets_3d_sign(&mut pw, codebooks_targets, codebooks)?;
    input_targets_2d_sign(&mut pw, ivf_center_targets, ivf_center)?;
    input_targets_1d(&mut pw, ivf_roots_targets, ivf_roots)?;
    input_targets_1d_sign(&mut pw, cluster_idxes_targets, cluster_idxes)?;
    input_targets_2d_sign(&mut pw, cluster_center_targets, cluster_center)?;
    input_targets_2d_sign(&mut pw, valids_targets, valids)?;
    input_targets_2d_sign(&mut pw, itemss_targets, itemss)?;
    input_targets_3d_sign(&mut pw, vpqss_targets, vpqss)?;
    input_targets_3d(&mut pw, cluster_pairs_targets, cluster_pairs)?;
    println!("输入witness: {:?}", curr_time.elapsed());

    let (build_time, prove_time, verify_time, proof_size, memory_used) = metrics_eval(builder, pw)?;
    Ok((build_time, prove_time, verify_time, proof_size, memory_used))
}
