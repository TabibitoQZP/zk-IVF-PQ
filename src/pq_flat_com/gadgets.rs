use crate::hash_gadgets::merkle_back_gadget;
use crate::pq_flat::gadgets::codebooks_query_gadget;
use crate::prelude::*;
use crate::utils::common_gadgets::sum_gadget;
use crate::utils::lookup::merkle_lut2d_gadget;
use crate::utils::set_gadgets::set_equal_gadget;

/*
* PQ-Flat要扫所有的量化节点
* D: 原始向量的维度, 这里写作D_, 因为有命名冲突
* M: 码本数量
* K: 簇的数量 (通常256)
* d=D/M
*/
pub fn pq_flat_com_gadget(
    builder: &mut CircuitBuilder<F, D>,      // builder
    fs_hash: Vec<Target>,                    // Fiat-Shamior用的值 (2,)
    codebooks: Vec<Vec<Vec<Target>>>,        // 全局码本 *(M,K,d)
    query: Vec<Target>,                      // 查询向量 (D,)
    pq_vecs: Vec<Vec<Target>>,               // 量化数据库, 未按距离序 *(N,M)
    pq_dis: Vec<Vec<Target>>,                // 前面量化对应的距离 (N,M)
    merkle_path: Vec<Vec<Vec<Vec<Target>>>>, // 各自的merkle路径 (N,M,m_d,2)
    sorted_idx_dis: Vec<Vec<Target>>,        // query对应pq向量的(idx,dis)对, 按dis非递减序 (N,2)
) {
    let D_ = query.len();
    let M = codebooks.len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();
    let N = pq_vecs.len();

    // 用码本计算和每个查询的距离 (M,K), 并计算merkle根root
    let lut = codebooks_query_gadget(builder, codebooks, query);
    let root = merkle_lut2d_gadget(builder, lut);

    // 证明pq_vecs, pq_dis, merkle_path符合条件
    let mut const_list: Vec<Target> = Vec::with_capacity(M);
    for i in 0..M {
        let item = builder.constant(F::from_canonical_u64(i as u64));
        builder.register_public_input(item); // 注册为公开
        const_list.push(item);
    }
    for i in 0..N {
        for j in 0..M {
            let curr_root = merkle_back_gadget(
                builder,
                vec![const_list[j].clone(), pq_vecs[i][j], pq_dis[i][j]],
                merkle_path[i][j].clone(),
            );
            builder.connect(curr_root, root);
        }
    }

    // 利用pq_dis累计和计算距离, 并判断集合与sorted_idx_dis一致
    let mut unsorted_idx_dis: Vec<Vec<Target>> = Vec::with_capacity(N);
    for i in 0..N {
        unsorted_idx_dis.push(vec![
            builder.constant(F::from_canonical_u64(i as u64)),
            sum_gadget(builder, pq_dis[i].clone()),
        ]);
    }
    set_equal_gadget(
        builder,
        fs_hash[0],
        fs_hash[1],
        unsorted_idx_dis,
        sorted_idx_dis.clone(),
    );

    // 排序校验
    for i in 0..N - 1 {
        let sub_target = builder.sub(sorted_idx_dis[i + 1][1], sorted_idx_dis[i][1]);
        builder.range_check(sub_target, 32);
    }
}
