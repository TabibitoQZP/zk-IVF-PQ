use crate::brute_force::gadgets::rev_gadget;
use crate::circuit_ivf_pq::gadgets::{dot_prod_gadget, one_hot_gadget};
use crate::hash_gadgets::merkle_back_gadget;
use crate::ivf_pq_verify::gadgets::vec_sub_gadget;
use crate::merkle_ver::ivf_pq_merkle::{
    commit_codebook_gadget, merkle_cluster_gadget, merkle_ivf_gadget,
};
use crate::merkle_ver::standalone_commitment::standalone_commitment_gadget;
use crate::pq_flat::gadgets::codebooks_query_gadget;
use crate::prelude::*;
use crate::utils::dis_gadgets::distance;
use crate::utils::nn_gadgets::comp_gadget;

pub fn circuit_based_ivf_pq_gadget(
    builder: &mut CircuitBuilder<F, D>,
    query: Vec<Target>,                       // 查询向量 (D,)
    top_k: usize,                             // 最终返回top_k个
    root: Target,                             // 总的根
    codebooks_root: Target,                   // codebooks对应的根
    codebooks: Vec<Vec<Vec<Target>>>,         // codebooks (M,K,d)
    mut ivf_center: Vec<Vec<Target>>,         // cluster的中心, 注意排序过程需要mut (n_list, D)
    ivf_roots: Vec<Target>,                   // cluster对应的merkle根 (n_list,)
    cluster_idxes: Vec<Target>,               // cluster对应的索引号 (n_probe,)
    cluster_center: Vec<Vec<Target>>,         // cluster中心也要提供 (n_probe,D)
    valids: Vec<Vec<Target>>,                 // 对应的vpqs是否valid (n_probe,n)
    itemss: Vec<Vec<Target>>,                 // 需要取出的内容 (n_probe,n)
    vpqss_onehot: Vec<Vec<Vec<Vec<Target>>>>, // 量化后的向量对应的one-hot (n_probe,n,M,K)
    cluster_pairs: Vec<Vec<Vec<Target>>>,     // merkle树的路径
                                              // vpqss: Vec<Vec<Vec<Target>>>,     // 量化后的向量 (n_probe,n,M)
) {
    // 基本变量
    let D_ = query.len();
    let M = codebooks.len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();
    let n_list = ivf_center.len();
    let n_probe = cluster_idxes.len();
    let n = valids[0].len();

    // 用 vpqss_onehot 组建vpqss
    let mut constant_targets: Vec<Target> = Vec::with_capacity(K);
    for i in 0..K {
        let curr_target = builder.constant(F::from_canonical_u64(i as u64));
        builder.register_public_input(curr_target);
        constant_targets.push(curr_target);
    }
    let mut vpqss: Vec<Vec<Vec<Target>>> = Vec::with_capacity(n_probe); // (n_probe, n, M)
    for i in 0..n_probe {
        let mut mat: Vec<Vec<Target>> = Vec::with_capacity(n);
        for j in 0..n {
            let mut row: Vec<Target> = Vec::with_capacity(M);
            for k in 0..M {
                one_hot_gadget(builder, vpqss_onehot[i][j][k].clone()); // 注意验一下one-hot
                let curr_idx = dot_prod_gadget(
                    builder,
                    vpqss_onehot[i][j][k].clone(),
                    constant_targets.clone(),
                );
                row.push(curr_idx);
            }
            mat.push(row);
        }
        vpqss.push(mat);
    }

    // 承诺部分
    standalone_commitment_gadget(
        builder,
        query.clone(),
        root.clone(),
        codebooks_root.clone(),
        codebooks.clone(),
        ivf_center.clone(),
        ivf_roots.clone(),
        cluster_idxes.clone(),
        cluster_center.clone(),
        valids.clone(),
        itemss.clone(),
        cluster_pairs.clone(),
        vpqss.clone(),
    );

    // 初始化索引和距离
    let mut idxes: Vec<Target> = (0..n_list)
        .map(|item| builder.constant(F::from_canonical_u64(item as u64)))
        .collect(); // (n_list,)
    let mut center_dis: Vec<Target> = ivf_center
        .clone()
        .into_iter()
        .map(|raw| distance(builder, raw, query.clone()))
        .collect(); // (n_list,)

    // 冒泡排序进行交换
    for i in 0..n_probe {
        for j in ((i + 1)..n_list).rev() {
            let comp_result = comp_gadget(builder, center_dis[j - 1], center_dis[j]);
            (center_dis[j - 1], center_dis[j]) =
                rev_gadget(builder, center_dis[j - 1], center_dis[j], comp_result);
            // 所有和距离相关的都要换
            (idxes[j - 1], idxes[j]) = rev_gadget(builder, idxes[j - 1], idxes[j], comp_result);
            // 换一下ivf_centers
            for k in 0..D_ {
                (ivf_center[j - 1][k], ivf_center[j][k]) = rev_gadget(
                    builder,
                    ivf_center[j - 1][k].clone(),
                    ivf_center[j][k].clone(),
                    comp_result,
                );
            }
        }
        // i次冒泡完成, i号索引就完成了排序, 验证是否一致
        builder.connect(idxes[i].clone(), cluster_idxes[i].clone());
    }

    // 打表
    let mut luts: Vec<Vec<Vec<Target>>> = Vec::with_capacity(n_probe as usize); // (n_probe,M,K)
    for i in 0..n_probe {
        let sub_val = vec_sub_gadget(builder, query.clone(), cluster_center[i].clone());
        luts.push(codebooks_query_gadget(builder, codebooks.clone(), sub_val));
    }

    // 开始计算cluster内的距离
    let all_count = n_probe * n;
    let mut cluster_dis: Vec<Target> = Vec::with_capacity(all_count);
    let mut flat_items: Vec<Target> = Vec::with_capacity(all_count);

    let one = builder.one();
    let max_gadget = builder.constant(F::from_canonical_u64(9223372036854775807)); // 2^63-1
    for i in 0..n_probe {
        for j in 0..n {
            let mut total_dis = builder.zero();
            for k in 0..M {
                let curr_one_hot = vpqss_onehot[i][j][k].clone();
                let curr_lut = luts[i][k].clone();
                let curr_dis = dot_prod_gadget(builder, curr_one_hot, curr_lut);
                total_dis = builder.add(total_dis, curr_dis);
            }
            let vld = valids[i][j];
            let sub_vld = builder.sub(one, vld);
            let vld_dis = builder.mul(vld, total_dis);
            let max_dis = builder.mul(sub_vld, max_gadget);
            total_dis = builder.add(vld_dis, max_dis);
            cluster_dis.push(total_dis);
            flat_items.push(itemss[i][j]);
        }
    }

    // 开始对cluster_dis排序, 并permute flat_items
    for i in 0..top_k {
        for j in ((i + 1)..all_count).rev() {
            let comp_result = comp_gadget(builder, cluster_dis[j - 1], cluster_dis[j]);
            (cluster_dis[j - 1], cluster_dis[j]) =
                rev_gadget(builder, cluster_dis[j - 1], cluster_dis[j], comp_result);
            // permute flat_items
            (flat_items[j - 1], flat_items[j]) =
                rev_gadget(builder, flat_items[j - 1], flat_items[j], comp_result);
        }
    }
}
