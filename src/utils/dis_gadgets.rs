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
