use crate::ivf_pq::gadgets::vec_sub_gadget;
use crate::ivf_pq_verify::gadgets::const_gen_gadget;
use crate::merkle_ver::standalone_commitment::standalone_commitment_gadget;
use crate::pq_flat::gadgets::codebooks_query_gadget;
use crate::pq_flat_verify::gadgets::set_belong_gedget;
use crate::prelude::*;
use crate::utils::dis_gadgets::distance;
use crate::utils::nn_gadgets::{comp_gadget, static_nn_gadget};
use crate::utils::set_gadgets::set_equal_gadget;
use std::cmp::max;

pub fn set_based_ivf_pq_gadget(
    builder: &mut CircuitBuilder<F, D>,
    fs_hash: Vec<Target>,                     // 挑战, (7,)
    query: Vec<Target>,                       // 查询向量 (D,)
    top_k: usize,                             // 最终返回top_k个
    root: Target,                             // 总的根
    codebooks_root: Target,                   // codebooks对应的根
    codebooks: Vec<Vec<Vec<Target>>>,         // codebooks (M,K,d)
    ivf_center: Vec<Vec<Target>>,             // cluster的中心, 注意排序过程需要mut (n_list, D)
    ivf_roots: Vec<Target>,                   // cluster对应的merkle根 (n_list,)
    cluster_center: Vec<Vec<Target>>,         // cluster中心也要提供 (n_probe,D)
    valids: Vec<Vec<Target>>,                 // 对应的vpqs是否valid (n_probe,n)
    itemss: Vec<Vec<Target>>,                 // 需要取出的内容 (n_probe,n)
    cluster_pairs: Vec<Vec<Vec<Target>>>,     // merkle树的路径
    vpqss: Vec<Vec<Vec<Target>>>,             // 量化后的向量 (n_probe,n,M)
    vpqss_dis: Vec<Vec<Vec<Target>>>,         // vpqss中的每一个索引号对应的LUT距离 (n_probe,n,M)
    ordered_vpqss_item_dis: Vec<Vec<Target>>, // vpqss中计算的距离和item集合 (n_probe*n,2)
    cluster_idx_dis: Vec<Vec<Target>>,        // cluster对应的索引号以及距离 (n_list,2)
    f_: Vec<Target>,
    t_: Vec<Target>,
) {
    let M = codebooks.len();
    let K = codebooks[0].len();
    let n = valids[0].len();
    let n_probe = cluster_center.len();
    let cluster_idxes: Vec<Target> = (0..n_probe)
        .map(|i| cluster_idx_dis[i][0].clone())
        .collect();
    let const_list = const_gen_gadget(
        builder,
        max(max(n_probe as u64, n as u64), max(M as u64, K as u64)),
    );

    // 计算距离
    let ivf_dis: Vec<Target> = ivf_center
        .clone()
        .into_iter()
        .map(|item| distance(builder, item, query.clone()))
        .collect();

    // ivf过程的证明
    static_nn_gadget(
        builder,
        fs_hash[0],
        fs_hash[1],
        ivf_center.clone(),
        query.clone(),
        cluster_idx_dis.clone(),
    );

    // # pq过程证明
    // 1. 打表
    let mut luts: Vec<Vec<Vec<Target>>> = Vec::with_capacity(n_probe); // (n_probe,M,K)
    for i in 0..n_probe {
        let sub_val = vec_sub_gadget(builder, query.clone(), cluster_center[i].clone());
        luts.push(codebooks_query_gadget(builder, codebooks.clone(), sub_val));
    }
    // 2. 构建lut_set
    let mut lut_set: Vec<Vec<Target>> = Vec::with_capacity(n_probe * M * K);
    for i in 0..n_probe {
        for j in 0..M {
            for k in 0..K {
                lut_set.push(vec![
                    const_list[i],
                    const_list[j],
                    const_list[k],
                    luts[i][j][k],
                ]);
            }
        }
    }
    // 3. 构建vpqs_set, 顺便计算距离
    let max_gadget = builder.constant(F::from_canonical_u64(9223372036854775807)); // 2^63-1
    let mut vpqss_item_dis: Vec<Vec<Target>> = Vec::with_capacity(n_probe * n); // (n_probe*n,2)
    let mut vpqss_set: Vec<Vec<Target>> = Vec::with_capacity(n_probe * n * M);
    for i in 0..n_probe {
        for j in 0..n {
            for k in 0..M {
                vpqss_set.push(vec![
                    const_list[i],
                    const_list[k],
                    vpqss[i][j][k],
                    vpqss_dis[i][j][k],
                ]);
            }
            let curr_dis = builder.add_many(vpqss_dis[i][j].clone());
            let vld = valids[i][j];
            let sub_vld = builder.sub(const_list[1].clone(), vld);
            let vld_dis = builder.mul(vld, curr_dis);
            let max_dis = builder.mul(sub_vld, max_gadget);
            vpqss_item_dis.push(vec![itemss[i][j], builder.add(vld_dis, max_dis)]);
        }
    }
    // 4. 校验集合相等, 并验证满足非递减序
    set_equal_gadget(
        builder,
        fs_hash[2],
        fs_hash[3],
        vpqss_item_dis,
        ordered_vpqss_item_dis.clone(),
    );
    for i in 0..(n_probe * n - 1) {
        let flag = comp_gadget(
            builder,
            ordered_vpqss_item_dis[i][1].clone(),
            ordered_vpqss_item_dis[i + 1][1].clone(),
        );
        builder.connect(flag, const_list[0]);
    }
    // 5. 验证 vpqss_set \subset lut_set
    set_belong_gedget(builder, fs_hash[4..].to_vec(), vpqss_set, lut_set, f_, t_);
    // 6. 将top_k对应的item释放
    for i in 0..top_k {
        builder.register_public_input(ordered_vpqss_item_dis[i][0]);
    }

    // 承诺
    standalone_commitment_gadget(
        builder,
        query.clone(),
        root.clone(),
        codebooks_root.clone(),
        codebooks.clone(),
        ivf_center.clone(),
        ivf_roots.clone(),
        cluster_idxes,
        cluster_center.clone(),
        valids.clone(),
        itemss.clone(),
        cluster_pairs.clone(),
        vpqss.clone(),
    );
}
