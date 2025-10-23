use crate::prelude::*;

pub fn l2(builder: &mut CircuitBuilder<F, D>, a: Target, b: Target) -> Target {
    let delta = builder.sub(a, b);
    builder.square(delta)
}

pub fn distance(builder: &mut CircuitBuilder<F, D>, src: Vec<Target>, dst: Vec<Target>) -> Target {
    let d = src.len();
    let mut cur_target = l2(builder, src[0], dst[0]);
    for i in 1..d {
        let tmp_target = l2(builder, src[i], dst[i]);
        cur_target = builder.add(cur_target, tmp_target);
    }
    cur_target
}

// 集合相等的验证电路, 注意F-S变换值要从外面拿
pub fn set_equal_gadget(
    builder: &mut CircuitBuilder<F, D>,
    r: Target,
    t: Target,
    src: Vec<Vec<Target>>,
    dst: Vec<Vec<Target>>,
) {
    /*
     * r: F-S challenge
     * t: F-S hash value, used to prove permutation
     * n: set size
     * d: set element dimension
     */
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
