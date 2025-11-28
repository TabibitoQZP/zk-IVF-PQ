use crate::hash_gadgets::{hash_tree_gen, hash_tree_path, tree_depth};
use crate::merkle_ver::circuit_based::circuit_based_ivf_pq_gadget;
use crate::merkle_ver::ivf_pq_merkle::{commit_codebook_i64, merkle_cluster_i64};
use crate::prelude::*;
use crate::utils::metrics::metrics_eval;

pub fn one_hot_gen_3d(origin: Vec<Vec<Vec<i64>>>, sz: usize) -> Vec<Vec<Vec<Vec<i64>>>> {
    let x = origin.len();
    let y = origin[0].len();
    let z = origin[0][0].len();
    let mut one_hot: Vec<Vec<Vec<Vec<i64>>>> = Vec::with_capacity(x);
    for i in 0..x {
        let mut cube: Vec<Vec<Vec<i64>>> = Vec::with_capacity(y);
        for j in 0..y {
            let mut mat: Vec<Vec<i64>> = Vec::with_capacity(z);
            for k in 0..z {
                let mut row: Vec<i64> = Vec::with_capacity(sz);
                for t in 0..sz {
                    if t as i64 == origin[i][j][k] {
                        row.push(1);
                    } else {
                        row.push(0);
                    }
                }
                mat.push(row);
            }
            cube.push(mat);
        }
        one_hot.push(cube);
    }
    one_hot
}

pub fn hash_ivf_center(i: usize, c_i: Vec<i64>, root_i: u64) -> u64 {
    let mut row: Vec<F> = Vec::with_capacity(c_i.len() + 2);
    let c_i_F: Vec<F> = c_i.into_iter().map(F::from_noncanonical_i64).collect();
    row.push(F::from_canonical_u64(i as u64));
    row.extend(c_i_F);
    row.push(F::from_canonical_u64(root_i));
    PoseidonHash::hash_no_pad(&row).elements[0].to_canonical_u64()
}

pub fn circuit_based_ivf_pq_proof(
    query: Vec<i64>,               // 查询向量 (D,)
    mut ivf_center: Vec<Vec<i64>>, // ivf簇中心 (n_list,D)
    cluster_idxes: Vec<i64>,       // 簇索引 (n_probe,)
    vpqss: Vec<Vec<Vec<i64>>>,     // 这里给原始向量, 手动改one-hot (n_probe,n,M)
    valids: Vec<Vec<i64>>,         // vpqss中向量是否valid (n_probe,n)
    itemss: Vec<Vec<i64>>,         // vpqss中向量对应的查询量 (n_probe,n)
    codebooks: Vec<Vec<Vec<i64>>>, // 全局码本 (M,K,d)
    ivf_roots: Vec<u64>,           // 这里给一下ivf各个root, 用来手算和还原数据 (n_list,)
    top_k: i64,                    // 明确取哪top_k
) -> Result<(f64, f64, f64, u64, u64), Box<dyn std::error::Error>> {
    // 基础变量
    let D_ = query.len();
    let M = codebooks.len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();
    let n_list = ivf_center.len();
    let n_probe = vpqss.len();
    let n = vpqss[0].len();

    let vpqss_onehot = one_hot_gen_3d(vpqss.clone(), K);
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
    let vpqss_onehot_targets = add_targets_4d(&mut builder, vec![n_probe, n, M, K]);
    let cluster_pairs_targets = add_targets_3d(&mut builder, vec![n_probe, depth, 2]);

    circuit_based_ivf_pq_gadget(
        &mut builder,
        query_targets.clone(),
        top_k as usize,
        root_targets.clone(),
        codebooks_root_targets.clone(),
        codebooks_targets.clone(),
        ivf_center_targets.clone(),
        ivf_roots_targets.clone(),
        cluster_idxes_targets.clone(),
        cluster_center_targets.clone(),
        valids_targets.clone(),
        itemss_targets.clone(),
        vpqss_onehot_targets.clone(),
        cluster_pairs_targets.clone(),
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
    input_targets_4d_sign(&mut pw, vpqss_onehot_targets, vpqss_onehot)?;
    input_targets_3d(&mut pw, cluster_pairs_targets, cluster_pairs)?;
    println!("输入witness: {:?}", curr_time.elapsed());

    let (build_time, prove_time, verify_time, proof_size, memory_used) = metrics_eval(builder, pw)?;
    Ok((build_time, prove_time, verify_time, proof_size, memory_used))
}
