use crate::prelude::*;

pub fn compress_gadget(
    builder: &mut CircuitBuilder<F, D>,
    alpha: Target,
    row: Vec<Target>,
) -> Target {
    let mut curr_target = row[0];
    for i in 1..row.len() {
        curr_target = builder.mul(alpha, curr_target);
        curr_target = builder.add(row[i], curr_target);
    }
    curr_target
}

pub fn simple_set_equal_gadget(
    builder: &mut CircuitBuilder<F, D>,
    r: Target,
    src: Vec<Target>,
    dst: Vec<Target>,
) {
    let fac_src: Vec<Target> = src.into_iter().map(|item| builder.sub(r, item)).collect();
    let fac_dst: Vec<Target> = dst.into_iter().map(|item| builder.sub(r, item)).collect();

    let prod_src = builder.mul_many(fac_src);
    let prod_dst = builder.mul_many(fac_dst);

    builder.connect(prod_src, prod_dst);
}

/*
* 证明\prod(r-\sum t^is_i)=\prod(r-\sum t^id_i)
*/
pub fn set_equal_gadget(
    builder: &mut CircuitBuilder<F, D>,
    r: Target,
    t: Target,
    src: Vec<Vec<Target>>,
    dst: Vec<Vec<Target>>,
) {
    let com_src: Vec<Target> = src
        .into_iter()
        .map(|row| compress_gadget(builder, t, row))
        .collect();
    let com_dst: Vec<Target> = dst
        .into_iter()
        .map(|row| compress_gadget(builder, t, row))
        .collect();
    simple_set_equal_gadget(builder, r, com_src, com_dst);
}
