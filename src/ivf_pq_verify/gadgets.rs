use crate::pq_flat::gadgets::codebooks_query_gadget;
use crate::pq_flat_verify::gadgets::set_belong_gedget;
use crate::prelude::*;
use crate::utils::common_gadgets::static_lookup_gadget;
use crate::utils::nn_gadgets::{comp_gadget, static_nn_gadget};
use std::cmp::max;

pub fn const_gen_gadget(builder: &mut CircuitBuilder<F, D>, n: u64) -> Vec<Target> {
    let targets: Vec<Target> = (0..n)
        .map(|item| builder.constant(F::from_canonical_u64(item)))
        .collect();
    targets
}

pub fn vec_sub_gadget(
    builder: &mut CircuitBuilder<F, D>, // builder
    a: Vec<Target>,
    b: Vec<Target>,
) -> Vec<Target> {
    let n = a.len();
    let mut result: Vec<Target> = Vec::with_capacity(n);

    for i in 0..n {
        result.push(builder.sub(a[i], b[i]));
    }
    result
}

/*
* 这里需要注意, 考虑到每个cluster内的向量元素不定, 所以我们需要设置一个上限max_sz,
* 我们只考虑不足max_sz的情况
*/
pub fn ivf_pq_verify_gadget(
    builder: &mut CircuitBuilder<F, D>, // builder
    fs_hash: Vec<Target>,               // Fiat-Shamior用的值 (5,)
    ivf_centers: Vec<Vec<Target>>,      // ivf簇中心 *(n_list,D)
    query: Vec<Target>,                 // 查询向量 (D,)
    sorted_idx_dis: Vec<Vec<Target>>,   // query对应簇中心(idx,dis)对, 按dis非递减序 (n_list,2)
    filtered_centers: Vec<Vec<Target>>, // 排序后距离小的簇中心 *(n_probe,D)
    probe_count: Vec<Target>,           // n_probe个簇内部所拥有的vec个数 (n_probe,)
    filtered_vecs: Vec<Vec<Target>>,    // n_probe个簇对应的所有向量, 已经排序好了 (max_sz,M)
    filtered_dis: Vec<Vec<Target>>,     // n_probe个簇对应的分量距离, 和前面的vec对应(max_sz,M)
    vecs_cluster_hot: Vec<Vec<Target>>, // 所有向量对应的簇索引, 表示为one-hot形式 (max_sz,n_probe)
    codebooks: Vec<Vec<Vec<Target>>>,   // 全局码本 (M,K,d)
    f_: Vec<Target>,
    t_: Vec<Target>,
) {
    let n_probe = probe_count.len();
    let max_sz = filtered_vecs.len();
    let M = codebooks.len();
    let K = codebooks[0].len();

    let const_list = const_gen_gadget(
        builder,
        max(max(n_probe as u64, max_sz as u64), max(M as u64, K as u64)),
    );

    // 验证距离计算以及排序是否正确
    static_nn_gadget(
        builder,
        fs_hash[0],
        fs_hash[1],
        ivf_centers,
        query.clone(),
        sorted_idx_dis.clone(),
    );

    let one = builder.one();

    // # vecs_cluster_hot的证明
    // 1. 约束0, 1
    for i in 0..max_sz {
        for j in 0..n_probe {
            static_lookup_gadget(builder, vecs_cluster_hot[i][j].clone(), vec![0, 1]);
        }
    }
    // 2. 计算水平和垂直求和
    let mut hor_sum: Vec<Target> = Vec::with_capacity(max_sz); // 水平求和
    let mut ver_sum: Vec<Target> = Vec::with_capacity(n_probe); // 垂直求和
    for i in 0..max_sz {
        let mut cur_target = builder.zero();
        for j in 0..n_probe {
            cur_target = builder.add(cur_target, vecs_cluster_hot[i][j].clone());
        }
        hor_sum.push(cur_target);
    }
    for j in 0..n_probe {
        let mut cur_target = builder.zero();
        for i in 0..max_sz {
            cur_target = builder.add(cur_target, vecs_cluster_hot[i][j].clone());
        }
        ver_sum.push(cur_target);
    }
    // 3. 水平求和只能是0/1
    for i in 0..max_sz {
        static_lookup_gadget(builder, hor_sum[i], vec![0, 1]);
    }
    // 4. 垂直求和要和probe簇内值一样
    for i in 0..n_probe {
        builder.connect(ver_sum[i], probe_count[i]);
    }

    // # 计算距离
    // 1. 计算luts并打表
    let mut luts: Vec<Vec<Vec<Target>>> = Vec::with_capacity(n_probe); // (n_probe,M,K)
    for i in 0..n_probe {
        let sub_val = vec_sub_gadget(builder, query.clone(), filtered_centers[i].clone());
        luts.push(codebooks_query_gadget(builder, codebooks.clone(), sub_val));
    }
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
    // 2. 计算距离并打表
    let max_gadget = builder.constant(F::from_canonical_u64(9223372036854775807)); // 2^63-1
    let mut dis_vec: Vec<Target> = Vec::with_capacity(max_sz);
    let mut dis_set: Vec<Vec<Target>> = Vec::with_capacity(max_sz);
    for i in 0..max_sz {
        let dis = builder.add_many(filtered_dis[i].clone());
        // 将元素进行打表
        for j in 0..M {
            // 计算当前位置的n_probe索引
            let mut lut_idx = const_list[0].clone();
            for k in 0..n_probe {
                lut_idx = builder.mul_add(
                    vecs_cluster_hot[i][k].clone(),
                    const_list[k].clone(),
                    lut_idx,
                );
            }
            // 打表
            dis_set.push(vec![
                lut_idx,
                const_list[j],
                filtered_vecs[i][j],
                filtered_dis[i][j],
            ]);
        }
        let cur_b = hor_sum[i];
        let sub_cur_b = builder.sub(one, cur_b);
        let cur_part = builder.mul(cur_b, dis);
        let sub_cur_part = builder.mul(sub_cur_b, max_gadget);
        let cur_dis = builder.add(cur_part, sub_cur_part);
        dis_vec.push(cur_dis);
    }
    // 3. 验证合法性
    set_belong_gedget(builder, fs_hash[2..].to_vec(), dis_set, lut_set, f_, t_);

    let zero = builder.zero();
    for i in 0..max_sz - 1 {
        // let sub_gadget = builder.sub(dis_vec[i + 1].clone(), dis_vec[i].clone());
        // builder.range_check(sub_gadget, 40);
        let tmp_target = comp_gadget(builder, dis_vec[i].clone(), dis_vec[i + 1].clone());
        builder.connect(tmp_target, zero);
    }
}
