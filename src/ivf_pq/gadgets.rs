use crate::pq_flat::gadgets::{codebooks_query_gadget, lut_code_gadget};
use crate::prelude::*;
use crate::utils::common_gadgets::static_lookup_gadget;
use crate::utils::dis_gadgets::distance;
use crate::utils::nn_gadgets::static_nn_gadget;

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

pub fn lut_choose_gadget(
    builder: &mut CircuitBuilder<F, D>, // builder
    luts: Vec<Vec<Vec<Target>>>,        // (n_probe,M,K)
    one_hot: Vec<Target>,               // (n_probe,)
) -> Vec<Vec<Target>> {
    let n_probe = luts.len();
    let M = luts[0].len();
    let K = luts[0][0].len();

    let mut lut: Vec<Vec<Target>> = Vec::with_capacity(M);
    for i in 0..M {
        let mut lut_row: Vec<Target> = Vec::with_capacity(K);
        for j in 0..K {
            let mut cur_cell = builder.zero();
            for k in 0..n_probe {
                let hot_val = builder.mul(one_hot[k].clone(), luts[k][i][j].clone());
                cur_cell = builder.add(cur_cell, hot_val);
            }
            lut_row.push(cur_cell);
        }
        lut.push(lut_row);
    }
    lut
}

/*
* 这里需要注意, 考虑到每个cluster内的向量元素不定, 所以我们需要设置一个上限max_sz,
* 我们只考虑不足max_sz的情况
*/
pub fn ivf_pq_gadget(
    builder: &mut CircuitBuilder<F, D>, // builder
    fs_hash: Vec<Target>,               // Fiat-Shamior用的值 (2,)
    ivf_centers: Vec<Vec<Target>>,      // ivf簇中心 *(n_list,D)
    query: Vec<Target>,                 // 查询向量 (D,)
    sorted_idx_dis: Vec<Vec<Target>>,   // query对应簇中心(idx,dis)对, 按dis非递减序 (n_list,2)
    filtered_centers: Vec<Vec<Target>>, // 排序后距离小的簇中心 *(n_probe,D)
    probe_count: Vec<Target>,           // n_probe个簇内部所拥有的vec个数 (n_probe,)
    filtered_vecs: Vec<Vec<Target>>,    // n_probe个簇对应的所有向量, 已经排序好了 (max_sz,M)
    vecs_cluster_hot: Vec<Vec<Target>>, // 所有向量对应的簇索引, 表示为one-hot形式 (max_sz,n_probe)
    codebooks: Vec<Vec<Vec<Target>>>,   // 全局码本 (M,K,d)
) {
    let n_probe = probe_count.len();
    let max_sz = filtered_vecs.len();

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
    // 1. 计算luts
    let mut luts: Vec<Vec<Vec<Target>>> = Vec::with_capacity(n_probe);
    for i in 0..n_probe {
        let sub_val = vec_sub_gadget(builder, query.clone(), filtered_centers[i].clone());
        luts.push(codebooks_query_gadget(builder, codebooks.clone(), sub_val));
    }
    // 2. 计算距离
    let max_gadget = builder.constant(F::from_canonical_u64(u32::MAX as u64));
    let mut dis_vec: Vec<Target> = Vec::with_capacity(max_sz);
    for i in 0..max_sz {
        let lut = lut_choose_gadget(builder, luts.clone(), vecs_cluster_hot[i].clone());
        let dis = lut_code_gadget(builder, lut, filtered_vecs[i].clone());
        let cur_b = hor_sum[i];
        let sub_cur_b = builder.sub(one, cur_b);
        let cur_part = builder.mul(cur_b, dis);
        let sub_cur_part = builder.mul(sub_cur_b, max_gadget);
        let cur_dis = builder.add(cur_part, sub_cur_part);
        dis_vec.push(cur_dis);
    }

    for i in 0..max_sz - 1 {
        let sub_gadget = builder.sub(dis_vec[i + 1].clone(), dis_vec[i].clone());
        builder.range_check(sub_gadget, 32);
    }
}
