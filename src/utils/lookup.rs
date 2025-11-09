use crate::hash_gadgets::merkle_tree_gadget;
use crate::prelude::*;

/*
* 将1/2/3维的LUT表转换成merkle根电路, 后续用root进行检验
* 检验方式是调用merkle_back_gadget, 提供索引和另一端即可
*/
pub fn merkle_lut1d_gadget(builder: &mut CircuitBuilder<F, D>, lut: Vec<Target>) -> Target {
    let x = lut.len();
    let mut leaves: Vec<Vec<Target>> = Vec::with_capacity(x);
    for i in 0..x {
        leaves.push(vec![
            builder.constant(F::from_canonical_u64(i as u64)),
            lut[i],
        ]);
    }
    merkle_tree_gadget(builder, leaves)
}
pub fn merkle_lut2d_gadget(builder: &mut CircuitBuilder<F, D>, lut: Vec<Vec<Target>>) -> Target {
    let x = lut.len();
    let y = lut[0].len();
    let mut leaves: Vec<Vec<Target>> = Vec::with_capacity(x);
    for i in 0..x {
        for j in 0..y {
            leaves.push(vec![
                builder.constant(F::from_canonical_u64(i as u64)),
                builder.constant(F::from_canonical_u64(j as u64)),
                lut[i][j],
            ]);
        }
    }
    merkle_tree_gadget(builder, leaves)
}
pub fn merkle_lut3d_gadget(
    builder: &mut CircuitBuilder<F, D>,
    lut: Vec<Vec<Vec<Target>>>,
) -> Target {
    let x = lut.len();
    let y = lut[0].len();
    let z = lut[0][0].len();
    let mut leaves: Vec<Vec<Target>> = Vec::with_capacity(x);
    for i in 0..x {
        for j in 0..y {
            for k in 0..z {
                leaves.push(vec![
                    builder.constant(F::from_canonical_u64(i as u64)),
                    builder.constant(F::from_canonical_u64(j as u64)),
                    builder.constant(F::from_canonical_u64(k as u64)),
                    lut[i][j][k],
                ]);
            }
        }
    }
    merkle_tree_gadget(builder, leaves)
}
