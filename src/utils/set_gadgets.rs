use crate::prelude::*;

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
    let n = src.len();
    let d = src[0].len();

    let mut fac_src: Vec<Target> = Vec::with_capacity(n);
    let mut fac_dst: Vec<Target> = Vec::with_capacity(n);
    for i in 0..n {
        let mut cur_targets = src[i][0];
        for j in 1..d {
            cur_targets = builder.mul(t, cur_targets);
            cur_targets = builder.add(cur_targets, src[i][j]);
        }
        fac_src.push(builder.sub(r, cur_targets));

        let mut cur_targetd = dst[i][0];
        for j in 1..d {
            cur_targetd = builder.mul(t, cur_targetd);
            cur_targetd = builder.add(cur_targetd, dst[i][j]);
        }
        fac_dst.push(builder.sub(r, cur_targetd));
    }

    let prod_src = builder.mul_many(fac_src);
    let prod_dst = builder.mul_many(fac_dst);

    builder.connect(prod_src, prod_dst);
}
