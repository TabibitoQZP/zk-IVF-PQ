use crate::ivfpq::utils::*;
use crate::prelude::*;

// 假定已经通过前面的nn获得了(idx, dis)
pub fn pg_gadget(
    builder: &mut CircuitBuilder<F, D>, // builder
    probe_idx: Vec<Target>,             // n_probe个center对应的索引
    probe_center: Vec<Vec<Target>>,     //n_probe个center对应的坐标
    codebooks: Vec<Vec<Vec<Target>>>,   // 公用codebooks
    query: Vec<Target>,                 // 查询向量
    codes: Vec<Vec<Target>>,            // 查询向量在给定n_probe个center下不同的codes
    codes_idx: Vec<Target>, // 注意这里的idx是lut中的idx, 不是n_list中的, n_list中的要probe_idx[idx]取回
    codes_origin_idx: Vec<Vec<Target>>, // 是真实的(own_idx, cluster_idx)对
) {
    // TODO: 验证probe_idx, probe_center确实是配对的 (可能要传入merkle路径)
    // 验证codes相关的正确性 (同样可能需要传入merkle路径)

    // 计算LUT表
    let lut = lut_gadget(builder, probe_center, codebooks, query);

    let l2_list = l2_lut_gadget(builder, lut, codes, codes_idx.clone());
    // 验证是否有序
    for i in 0..l2_list.len() {
        let sub_target = builder.sub(l2_list[i + 1].clone(), l2_list[i].clone());
        builder.range_check(sub_target, 32);
    }

    // 验证传入codes的可靠性
    for i in 0..codes_idx.len() {
        // 码本索引和真实索引是否一致
        let real_idx = builder.random_access(codes_idx[i], probe_idx.clone());
        builder.connect(real_idx, codes_origin_idx[i][1]);
    }

    // TODO: 别忘了各种merkle约束
}
