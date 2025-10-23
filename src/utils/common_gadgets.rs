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
