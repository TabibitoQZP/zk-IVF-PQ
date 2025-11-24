use crate::prelude::*;
use crate::utils::dis_gadgets::distance;
use crate::utils::set_gadgets::set_equal_gadget;

// 给定(src, dst), src<=dst返回0, src>dst返回1
pub fn comp_gadget(builder: &mut CircuitBuilder<F, D>, src: Target, dst: Target) -> Target {
    let sub_target = builder.sub(dst, src);
    let final_target = builder.split_le(sub_target, 64)[63];
    final_target.target
}

pub fn dynamic_nn_gadget(
    builder: &mut CircuitBuilder<F, D>, // builder
    query: Vec<Target>,                 // 查询向量
    src_vecs: Vec<Vec<Target>>,         // 距离计算向量, 这里已经针对query排好序了
) {
    let zero = builder.zero();
    builder.register_public_input(zero);
    let n = src_vecs.len();
    let mut dis_gadgets: Vec<Target> = Vec::with_capacity(src_vecs.len());

    // 计算距离
    for i in 0..n {
        dis_gadgets.push(distance(builder, query.clone(), src_vecs[i].clone()));
    }

    // 检查是否排序正确
    for i in 0..n - 1 {
        let flag = comp_gadget(builder, dis_gadgets[i].clone(), dis_gadgets[i + 1].clone());
        builder.connect(flag, zero);

        // let sub_gadget = builder.sub(dis_gadgets[i + 1].clone(), dis_gadgets[i].clone());
        // builder.range_check(sub_gadget, 32);
    }
}

// 这个是针对src_vecs写死承诺的情况
pub fn static_nn_gadget(
    builder: &mut CircuitBuilder<F, D>,
    r: Target,
    t: Target,
    src_vecs: Vec<Vec<Target>>,
    query: Vec<Target>,
    sorted_idx_dis: Vec<Vec<Target>>,
) {
    let zero = builder.zero();
    builder.register_public_input(zero);
    let n = src_vecs.len();

    // 计算未排序的(idx, dis) 对
    let mut unsorted_idx_dis: Vec<Vec<Target>> = Vec::with_capacity(n);
    for i in 0..n {
        unsorted_idx_dis.push(vec![
            builder.constant(F::from_canonical_u64(i as u64)),
            distance(builder, query.clone(), src_vecs[i].clone()),
        ])
    }
    set_equal_gadget(builder, r, t, unsorted_idx_dis, sorted_idx_dis.clone());

    // 验证排序是否是递增序
    for i in 0..n - 1 {
        let flag = comp_gadget(
            builder,
            sorted_idx_dis[i][1].clone(),
            sorted_idx_dis[i + 1][1].clone(),
        );
        builder.connect(flag, zero);
        // let tmp_target = builder.sub(
        //     sorted_idx_dis[i + 1][1].clone(),
        //     sorted_idx_dis[i][1].clone(),
        // );
        // builder.range_check(tmp_target, 32);
    }
}
