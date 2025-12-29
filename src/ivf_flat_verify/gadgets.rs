use crate::prelude::*;
use crate::utils::common_gadgets::static_lookup_gadget;
use crate::utils::dis_gadgets::distance;
use crate::utils::nn_gadgets::{comp_gadget, static_nn_gadget};
use crate::utils::set_gadgets::set_equal_gadget;

pub fn ivf_flat_verify_gadget(
    builder: &mut CircuitBuilder<F, D>,  // builder
    fs_hash: Vec<Target>,                // Fiat-Shamior用的值 (4,)
    ivf_centers: Vec<Vec<Target>>,       // ivf簇中心 (n_list,d)
    query: Vec<Target>,                  // 查询向量 (d,)
    sorted_idx_dis: Vec<Vec<Target>>,    // query对应簇中心(idx,dis)对, 按dis非递减序 (n_list,2)
    vecss: Vec<Vec<Vec<Target>>>,        // n_probe个簇对应的所有向量 (n_probe,n,d)
    valids: Vec<Vec<Target>>,            // vecss中向量是否valid (n_probe,n)
    itemss: Vec<Vec<Target>>,            // vecss中向量对应的item id (n_probe,n)
    ordered_items_dis: Vec<Vec<Target>>, // (n_probe*n,2), 全局按dis非递减序
    top_k: usize,                        // 明确取哪top_k
) {
    let n_probe = vecss.len();
    let n = valids[0].len();

    static_nn_gadget(
        builder,
        fs_hash[0],
        fs_hash[1],
        ivf_centers,
        query.clone(),
        sorted_idx_dis,
    );

    for i in 0..n_probe {
        for j in 0..n {
            static_lookup_gadget(builder, valids[i][j], vec![0, 1]);
        }
    }

    let one = builder.one();
    let max_dis = builder.constant(F::from_canonical_u64((1_u64 << 62) - 1));

    let mut items_dis: Vec<Vec<Target>> = Vec::with_capacity(n_probe * n);
    for i in 0..n_probe {
        for j in 0..n {
            let dis = distance(builder, query.clone(), vecss[i][j].clone());
            let vld = valids[i][j];
            let sub_vld = builder.sub(one, vld);
            let vld_dis = builder.mul(vld, dis);
            let inv_dis = builder.mul(sub_vld, max_dis);
            let final_dis = builder.add(vld_dis, inv_dis);
            items_dis.push(vec![itemss[i][j], final_dis]);
        }
    }

    set_equal_gadget(
        builder,
        fs_hash[2],
        fs_hash[3],
        items_dis,
        ordered_items_dis.clone(),
    );

    let zero = builder.zero();
    for i in 0..ordered_items_dis.len() - 1 {
        let flag = comp_gadget(
            builder,
            ordered_items_dis[i][1],
            ordered_items_dis[i + 1][1],
        );
        builder.connect(flag, zero);
    }

    for i in 0..top_k {
        builder.register_public_input(ordered_items_dis[i][0]);
    }
}
