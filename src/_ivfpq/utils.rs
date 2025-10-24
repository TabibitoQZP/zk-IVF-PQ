use crate::common_gadgets::distance;
use crate::hash_gadgets::{hash_gadget, merkle_tree_gadget};
use crate::prelude::*;

// 为x设置数个动态的约束
pub fn dynamic_lookup_gadget(builder: &mut CircuitBuilder<F, D>, x: Target, tb: Vec<Target>) {
    let mut cur_target = builder.one();
    let zero = builder.zero();
    for item in tb {
        let factor_target = builder.sub(x, item);
        cur_target = builder.mul(cur_target, factor_target);
    }
    builder.connect(cur_target, zero);
}

// 为x设置数个静态的约束
pub fn static_lookup_gadget(builder: &mut CircuitBuilder<F, D>, x: Target, tb: Vec<u64>) {
    let tb_target = tb
        .into_iter()
        .map(|x| builder.constant(F::from_canonical_u64(x)))
        .collect();
    dynamic_lookup_gadget(builder, x, tb_target);
}

// 生成基于leaf和hash链和方向的root
pub fn merkle_verify_gadget(
    builder: &mut CircuitBuilder<F, D>,
    leaf: Vec<Target>,
    hash_chain: Vec<Target>,
    hash_part: Vec<Target>,
) -> Target {
    let one = builder.one();
    let mut cur_hash = hash_gadget(builder, leaf);
    for i in 0..hash_part.len() {
        static_lookup_gadget(builder, hash_part[i], vec![0, 1]);
        let other_part = builder.sub(one, hash_part[0]);
        let v0 = builder.mul(cur_hash, hash_part[0]);
        let v1 = builder.mul(cur_hash, other_part);
        let v2 = builder.mul(hash_chain[i], hash_part[0]);
        let v3 = builder.mul(hash_chain[i], other_part);
        let left = builder.add(v0, v3);
        let right = builder.add(v1, v2);
        cur_hash = hash_gadget(builder, vec![left, right]);
    }
    cur_hash
}

// 电路中输入n_probe, 根据root进行信息验证
pub fn probe_verify_gadget(
    builder: &mut CircuitBuilder<F, D>,
    probe_idx: Vec<Target>,
    probe_center: Vec<Vec<Target>>,
    hash_chain: Vec<Vec<Target>>,
    hash_part: Vec<Vec<Target>>,
    root: Target,
) {
    /*
     * probe_idx: the n_probe closest cluster indexes
     * probe_center: the index refers center
     * hash_chain: the hash chain from every cluster to the root
     * root: merkle root of (idx, x_{idx})
     */
    let n_probe = probe_idx.len();
    let d = probe_center[0].len();

    // 证明输入的idx和center对应root
    for i in 0..n_probe {
        let mut tmp: Vec<Target> = Vec::with_capacity(d + 1);
        tmp.push(probe_idx[i]);
        tmp.extend(probe_center[i].clone());
        let cur_root =
            merkle_verify_gadget(builder, tmp, hash_chain[i].clone(), hash_part[i].clone());
        builder.connect(cur_root, root);
    }
}

// TODO: 实现codebook相关的验证
pub fn codebook_verify_gadget() {}

// 基于选取的簇索引和中心, 计算和codebooks各个索引的距离
pub fn lut_gadget(
    builder: &mut CircuitBuilder<F, D>,
    probe_center: Vec<Vec<Target>>,
    codebooks: Vec<Vec<Vec<Target>>>,
    query: Vec<Target>,
) -> Vec<Vec<Vec<Target>>> {
    let n_cb = codebooks.len(); // codebook数量, 通常是8/16
    let k_cb = codebooks[0].len(); // codebooks中簇数量, 通常为256
    let n_probe = probe_center.len();
    let d = probe_center[0].len();
    let d_cb = codebooks[0][0].len(); // 需要满足 d_cb * n_cb = d

    // 开始创建LUT
    let mut lut: Vec<Vec<Vec<Target>>> = Vec::with_capacity(n_probe);
    for i in 0..n_probe {
        // 将query拉到当前中心
        let cur_center = probe_center[i].clone();
        let mut cur_move_query: Vec<Target> = Vec::with_capacity(d);
        for j in 0..d {
            cur_move_query.push(builder.sub(query[j].clone(), cur_center[j].clone()));
        }

        let mut lut_i: Vec<Vec<Target>> = Vec::with_capacity(n_cb);
        for j in 0..n_cb {
            // 切片与当前codebook计算距离
            let cur_slide = cur_move_query[j * d_cb..(j + 1) * d_cb].to_vec();

            let mut lut_ij: Vec<Target> = Vec::with_capacity(k_cb);
            for k in 0..k_cb {
                // 与当前codebook的所有簇中心计算距离
                let cur_center_cb = codebooks[j][k].clone();
                let l2_dis = distance(builder, cur_center_cb, cur_slide.clone());
                lut_ij.push(l2_dis);
            }
            lut_i.push(lut_ij);
        }
        lut.push(lut_i);
    }
    lut // [索引号][码本号][码本k]
}

pub fn l2_lut_gadget(
    builder: &mut CircuitBuilder<F, D>,
    lut: Vec<Vec<Vec<Target>>>, // 比如 [n_list][n_cb][k_cb]
    codes: Vec<Vec<Target>>,
    codes_idx: Vec<Target>, // 注意这里的idx是lut中的idx, 不是n_list中的, n_list中的要probe_idx[idx]取回
) -> Vec<Target> {
    let n_probe = lut.len();
    let n_cb = lut[0].len();
    let k_cb = lut[0][0].len();
    let n_codes = codes.len();

    let mut plain_lut: Vec<Target> = Vec::with_capacity(n_probe * n_cb * k_cb);
    let k_cb_gadget = builder.constant(F::from_canonical_u64(k_cb as u64));
    let nk_cb_gadget = builder.constant(F::from_canonical_u64((n_cb * k_cb) as u64));

    for i in 0..n_probe {
        for j in 0..n_cb {
            for k in 0..k_cb {
                plain_lut.push(lut[i][j][k].clone());
            }
        }
    }

    let mut code_dis: Vec<Target> = Vec::with_capacity(n_codes);
    for i in 0..n_codes {
        // 开始计算每个code的距离
        let lut_i = codes_idx[i].clone(); // 取lut中的特定表
        let mut curr_l2 = builder.zero();
        for j in 0..n_cb {
            let lut_j = builder.constant(F::from_canonical_u64(j as u64));
            let lut_k = codes[i][j].clone();

            let add0 = builder.mul(lut_i, nk_cb_gadget.clone());
            let add1 = builder.mul(lut_j, k_cb_gadget.clone());
            let sum = builder.add_many(vec![add0, add1, lut_k]);
            let val = builder.random_access(sum, plain_lut.clone());
            curr_l2 = builder.add(curr_l2, val);
        }
        code_dis.push(curr_l2);
    }
    code_dis
}
