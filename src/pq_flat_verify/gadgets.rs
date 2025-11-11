use crate::prelude::*;
use crate::utils::dis_gadgets::distance;
use crate::utils::set_gadgets::{compress_gadget, set_equal_gadget, simple_set_equal_gadget};

// 计算LUT
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

pub fn set_belong_gedget(
    builder: &mut CircuitBuilder<F, D>,
    fs_hash: Vec<Target>,    // 随机挑战 (3,)
    f_set: Vec<Vec<Target>>, // 原始f集合
    t_set: Vec<Vec<Target>>, // 原始t集合
    f_: Vec<Target>,         // 外部计算的f
    t_: Vec<Target>,         // 外部计算的, 扩增的t
) {
    let f: Vec<Target> = f_set
        .into_iter()
        .map(|row| compress_gadget(builder, fs_hash[0], row))
        .collect();
    let mut t: Vec<Target> = t_set
        .into_iter()
        .map(|row| compress_gadget(builder, fs_hash[0], row))
        .collect();
    while t.len() < f.len() {
        t.push(t[0]);
    }

    simple_set_equal_gadget(builder, fs_hash[1], f.clone(), f_.clone());
    simple_set_equal_gadget(builder, fs_hash[2], t.clone(), t_.clone());

    let sz = f_.len();
    builder.connect(f_[0], t_[0]);
    let zero = builder.zero();
    for i in 1..sz {
        let left = builder.sub(f_[i], t_[i]);
        let right = builder.sub(f_[i], f_[i - 1]);
        let prod = builder.mul(left, right);
        builder.connect(prod, zero);
    }
}

/*
* PQ-Flat要扫所有的量化节点
* D: 原始向量的维度, 这里写作D_, 因为有命名冲突
* M: 码本数量
* K: 簇的数量 (通常256)
* d=D/M
*/
pub fn pq_flat_verify_gadget(
    builder: &mut CircuitBuilder<F, D>, // builder
    fs_hash: Vec<Target>,               // Fiat-Shamior用的值 (5,)
    codebooks: Vec<Vec<Vec<Target>>>,   // 全局码本 *(M,K,d)
    query: Vec<Target>,                 // 查询向量 (D,)
    pq_vecs: Vec<Vec<Target>>,          // 量化数据库, 未按距离序 *(N,M)
    pq_dis: Vec<Vec<Target>>,           // 对应前面vecs中的距离 (N,M)
    sorted_idx_dis: Vec<Vec<Target>>,   // query对应pq向量的(idx,dis)对, 按dis非递减序 (N,2)
    f_: Vec<Target>,                    // (N*M,)
    t_: Vec<Target>,                    // (N*M,)
) {
    let D_ = query.len();
    let M = codebooks.len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();
    let N = pq_vecs.len();

    // 用码本计算和每个查询的距离 (M,K)
    let lut = codebooks_query_gadget(builder, codebooks, query);

    // 直接用pq_dis计算距离
    let mut unsorted_idx_dis: Vec<Vec<Target>> = Vec::with_capacity(N);
    for i in 0..N {
        unsorted_idx_dis.push(vec![
            builder.constant(F::from_canonical_u64(i as u64)),
            builder.add_many(pq_dis[i].clone()),
        ]);
    }
    set_equal_gadget(
        builder,
        fs_hash[0],
        fs_hash[1],
        unsorted_idx_dis,
        sorted_idx_dis.clone(),
    );

    // 将LUT打造成集合T
    let mut t_set: Vec<Vec<Target>> = Vec::with_capacity(M * K);
    for i in 0..M {
        for j in 0..K {
            t_set.push(vec![
                builder.constant(F::from_canonical_u64(i as u64)),
                builder.constant(F::from_canonical_u64(j as u64)),
                lut[i][j],
            ])
        }
    }
    // 将pq_vecs打造成集合F
    let mut f_set: Vec<Vec<Target>> = Vec::with_capacity(N * M);
    for i in 0..N {
        for j in 0..M {
            f_set.push(vec![
                builder.constant(F::from_canonical_u64(j as u64)),
                pq_vecs[i][j],
                pq_dis[i][j],
            ])
        }
    }

    // 证明集合F和集合T相同
    set_belong_gedget(builder, fs_hash[2..].to_vec(), f_set, t_set, f_, t_);

    // 排序校验
    for i in 0..N - 1 {
        let sub_target = builder.sub(sorted_idx_dis[i + 1][1], sorted_idx_dis[i][1]);
        builder.range_check(sub_target, 32);
    }
}
