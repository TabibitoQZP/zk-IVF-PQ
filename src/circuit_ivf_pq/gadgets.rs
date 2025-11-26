use crate::brute_force::gadgets::rev_gadget;
use crate::ivf_pq_verify::gadgets::vec_sub_gadget;
use crate::pq_flat::gadgets::codebooks_query_gadget;
use crate::prelude::*;
use crate::utils::common_gadgets::static_lookup_gadget;
use crate::utils::dis_gadgets::distance;
use crate::utils::nn_gadgets::comp_gadget;

pub fn one_hot_gadget(builder: &mut CircuitBuilder<F, D>, row: Vec<Target>) {
    let one = builder.one();
    for item in row.clone() {
        static_lookup_gadget(builder, item, vec![0, 1]);
    }
    let sum_val = builder.add_many(row);
    builder.connect(sum_val, one);
}

pub fn dot_prod_gadget(
    builder: &mut CircuitBuilder<F, D>,
    a: Vec<Target>,
    b: Vec<Target>,
) -> Target {
    let sz = a.len();
    let mut sum_val = builder.zero();
    for i in 0..sz {
        let prod_target = builder.mul(a[i], b[i]);
        sum_val = builder.add(sum_val, prod_target);
    }
    sum_val
}

pub fn circuit_ivf_pq_gadget(
    builder: &mut CircuitBuilder<F, D>, // builder
    query: Vec<Target>,                 // 查询向量 (D,)
    mut ivf_centers: Vec<Vec<Target>>,  // ivf簇中心 *(n_list,D)
    vecs: Vec<Vec<Vec<Vec<Target>>>>,   // 这里每个都固定给到 (n_probe,max_sz,M,K)
    hot: Vec<Vec<Target>>,              // 针对vecs是否valid
    codebooks: Vec<Vec<Vec<Target>>>,   // 全局码本 (M,K,d)
    top_k: i64,                         // 明确取哪top_k
) {
    let n_list = ivf_centers.len();
    let D_ = ivf_centers[0].len();
    let n_probe = vecs.len();
    let max_sz = vecs[0].len();
    let M = codebooks.len();
    let K = codebooks[0].len();

    // 约束vecs全是one_hot的
    for cube in vecs.clone() {
        for mat in cube {
            for row in mat {
                one_hot_gadget(builder, row);
            }
        }
    }
    // 约束hot元素只能是0,1
    for row in hot.clone() {
        for item in row {
            static_lookup_gadget(builder, item, vec![0, 1]);
        }
    }

    // # 使用电路计算query与ivf_centers的距离, 并计算排序
    // 1. 生成center索引并计算距离
    let mut idxes: Vec<Target> = (0..n_list)
        .map(|item| builder.constant(F::from_canonical_u64(item as u64)))
        .collect(); // (n_list,)
    let mut center_dis: Vec<Target> = ivf_centers
        .clone()
        .into_iter()
        .map(|raw| distance(builder, raw, query.clone()))
        .collect(); // (n_list,)

    // 2. 计算n_probe次冒泡排序, 这样可以获得最小的前n_probe个值
    for i in 0..n_probe {
        for j in ((i + 1) as usize..n_list).rev() {
            let comp_result = comp_gadget(builder, center_dis[j - 1], center_dis[j]);
            (center_dis[j - 1], center_dis[j]) =
                rev_gadget(builder, center_dis[j - 1], center_dis[j], comp_result);
            // 所有和距离相关的都要换
            (idxes[j - 1], idxes[j]) = rev_gadget(builder, idxes[j - 1], idxes[j], comp_result);
            // 换一下ivf_centers
            for k in 0..D_ {
                (ivf_centers[j - 1][k], ivf_centers[j][k]) = rev_gadget(
                    builder,
                    ivf_centers[j - 1][k].clone(),
                    ivf_centers[j][k].clone(),
                    comp_result,
                );
            }
        }
    }
    // NOTE: 现在ivf_centers, idxes的前n_probe个有序了

    // 使用码本打表
    let mut luts: Vec<Vec<Vec<Target>>> = Vec::with_capacity(n_probe as usize); // (n_probe,M,K)
    for i in 0..n_probe {
        let sub_val = vec_sub_gadget(builder, query.clone(), ivf_centers[i as usize].clone());
        luts.push(codebooks_query_gadget(builder, codebooks.clone(), sub_val));
    }

    // 计算所有的距离
    let one = builder.one();
    let max_gadget = builder.constant(F::from_canonical_u64(9223372036854775807)); // 2^63-1
    let mut dis: Vec<Target> = Vec::with_capacity(n_probe * max_sz);
    for i in 0..n_probe {
        for j in 0..max_sz {
            let mut total_dis = builder.zero();
            for k in 0..M {
                let curr_one_hot = vecs[i][j][k].clone();
                let curr_lut = luts[i][k].clone();
                let curr_dis = dot_prod_gadget(builder, curr_one_hot, curr_lut);
                total_dis = builder.add(total_dis, curr_dis);
            }
            let vld = hot[i][j];
            let sub_vld = builder.sub(one, vld);
            let vld_dis = builder.mul(vld, total_dis);
            let max_dis = builder.mul(sub_vld, max_gadget);
            total_dis = builder.add(vld_dis, max_dis);
            dis.push(total_dis);
        }
    }

    // 排序
    for i in 0..top_k {
        for j in ((i + 1) as usize..dis.len()).rev() {
            let comp_result = comp_gadget(builder, dis[j - 1], dis[j]);
            (dis[j - 1], dis[j]) = rev_gadget(builder, dis[j - 1], dis[j], comp_result);
            // 所有和距离相关的都要换
        }
    }
}
