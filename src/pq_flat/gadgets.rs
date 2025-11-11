use crate::prelude::*;
use crate::utils::dis_gadgets::distance;
use crate::utils::set_gadgets::set_equal_gadget;

pub fn codebooks_query_gadget(
    builder: &mut CircuitBuilder<F, D>, // builder
    codebooks: Vec<Vec<Vec<Target>>>,   // 全局码本 （M,K,d)
    query: Vec<Target>,                 // 查询向量 (D,)
) -> Vec<Vec<Target>> // 要返回一个与各个码本的距离平方 (M,K)
{
    let D_ = query.len();
    let M = codebooks.len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();

    let mut result: Vec<Vec<Target>> = Vec::with_capacity(M);
    for i in 0..M {
        let mut cur_result: Vec<Target> = Vec::with_capacity(K);
        for j in 0..K {
            let cur_vec = codebooks[i][j].clone(); // (d,)
            let query_slide = query[(i * d)..((i + 1) * d)].to_vec();
            let dis = distance(builder, cur_vec, query_slide);
            cur_result.push(dis);
        }
        result.push(cur_result);
    }
    result
}

// 可以支持超过64长度的随机读取
pub fn random_access_gadget(
    builder: &mut CircuitBuilder<F, D>,
    idx: Target,
    data: Vec<Target>,
) -> Target {
    let sz = data.len();
    if sz <= 64 {
        return builder.random_access(idx, data);
    }
    let idx2d = builder.split_le_base::<8>(idx, 4);
    let const_8 = builder.constant(F::from_canonical_u64(8));
    let lower_part = builder.mul_add(idx2d[1], const_8, idx2d[0]);
    let higher_part = builder.mul_add(idx2d[3], const_8, idx2d[2]);

    let x = sz / 64;
    let data2d: Vec<Vec<Target>> = (0..x)
        .map(|i| data[i * 64..(i + 1) * 64].to_vec())
        .collect();

    let mut new_data: Vec<Target> = Vec::with_capacity(64);
    for i in 0..64 {
        let curr_row: Vec<Target> = (0..x).map(|y| data2d[y][i]).collect();
        new_data.push(builder.random_access(higher_part, curr_row));
    }
    builder.random_access(lower_part, new_data)
}

pub fn lut_code_gadget(
    builder: &mut CircuitBuilder<F, D>, // builder
    lut: Vec<Vec<Target>>,              // (M,K)
    code: Vec<Target>,                  // (M,)
) -> Target {
    // TODO: 将LUT分成多表来实现更复杂的支持
    let M = lut.len();
    let mut total_dis = builder.zero();

    for i in 0..M {
        let cur_dis = random_access_gadget(builder, code[i], lut[i].clone());
        total_dis = builder.add(total_dis, cur_dis);
    }
    total_dis
}

/*
* PQ-Flat要扫所有的量化节点
* D: 原始向量的维度, 这里写作D_, 因为有命名冲突
* M: 码本数量
* K: 簇的数量 (通常256)
* d=D/M
*/
pub fn pq_flat_gadget(
    builder: &mut CircuitBuilder<F, D>, // builder
    fs_hash: Vec<Target>,               // Fiat-Shamior用的值 (2,)
    codebooks: Vec<Vec<Vec<Target>>>,   // 全局码本 *(M,K,d)
    query: Vec<Target>,                 // 查询向量 (D,)
    pq_vecs: Vec<Vec<Target>>,          // 量化数据库, 未按距离序 *(N,M)
    sorted_idx_dis: Vec<Vec<Target>>,   // query对应pq向量的(idx,dis)对, 按dis非递减序 (N,2)
) {
    let D_ = query.len();
    let M = codebooks.len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();
    let N = pq_vecs.len();

    // 用码本计算和每个查询的距离 (M,K)
    let lut = codebooks_query_gadget(builder, codebooks, query);

    // 用lut查询和每个pq vec的距离
    let mut unsorted_idx_dis: Vec<Vec<Target>> = Vec::with_capacity(N);
    for i in 0..N {
        unsorted_idx_dis.push(vec![
            builder.constant(F::from_canonical_u64(i as u64)),
            lut_code_gadget(builder, lut.clone(), pq_vecs[i].clone()),
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
