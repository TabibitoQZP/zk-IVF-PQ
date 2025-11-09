// jemalloc使用, 但实测会出问题
// use jemallocator::Jemalloc;
//
// #[global_allocator]
// static GLOBAL: Jemalloc = Jemalloc;

// 只用于引入头文件
pub use anyhow::Result;
pub use plonky2::field::types::Field;
pub use plonky2::field::types::PrimeField64;
pub use plonky2::hash::poseidon::PoseidonHash;
pub use plonky2::iop::target::Target;
pub use plonky2::iop::witness::{PartialWitness, WitnessWrite};
pub use plonky2::plonk::circuit_builder::CircuitBuilder;
pub use plonky2::plonk::circuit_data::CircuitConfig;
pub use plonky2::plonk::config::Hasher;
pub use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
pub use std::result;
pub use std::time::Instant;

pub const D: usize = 2;
pub type C = PoseidonGoldilocksConfig;
pub type F = <C as GenericConfig<D>>::F;

pub fn make_builder() -> CircuitBuilder<F, D> {
    let cfg = CircuitConfig::standard_recursion_config();

    // 为了增大random access位数, 暂时不work
    // cfg.num_routed_wires = cfg.num_routed_wires.max(2 + 256); // >= 258
    // cfg.num_wires = cfg.num_wires.max(2 + 256 + 8); // >= 266
    // cfg.max_quotient_degree_factor = cfg.max_quotient_degree_factor.max(12); // 8~16 都可；12/16 更保险

    CircuitBuilder::<F, D>::new(cfg)
}

// public_targets系列
pub fn public_targets_1d(builder: &mut CircuitBuilder<F, D>, targets: Vec<Target>) {
    for t in targets {
        builder.register_public_input(t);
    }
}
pub fn public_targets_2d(builder: &mut CircuitBuilder<F, D>, targets: Vec<Vec<Target>>) {
    for t in targets {
        public_targets_1d(builder, t);
    }
}
pub fn public_targets_3d(builder: &mut CircuitBuilder<F, D>, targets: Vec<Vec<Vec<Target>>>) {
    for t in targets {
        public_targets_2d(builder, t);
    }
}

// add_targets系列
pub fn add_targets_2d(builder: &mut CircuitBuilder<F, D>, shape: Vec<usize>) -> Vec<Vec<Target>> {
    let mut result: Vec<Vec<Target>> = Vec::with_capacity(shape[0]);
    for _ in 0..shape[0] {
        result.push(builder.add_virtual_targets(shape[1]));
    }
    result
}
pub fn add_targets_3d(
    builder: &mut CircuitBuilder<F, D>,
    shape: Vec<usize>,
) -> Vec<Vec<Vec<Target>>> {
    let mut result: Vec<Vec<Vec<Target>>> = Vec::with_capacity(shape[0]);
    for _ in 0..shape[0] {
        result.push(add_targets_2d(builder, vec![shape[1], shape[2]]));
    }
    result
}

// input_targets系列
pub fn input_targets_1d(
    pw: &mut PartialWitness<F>,
    targets: Vec<Target>,
    inputs: Vec<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let x = targets.len();
    for i in 0..x {
        pw.set_target(targets[i], F::from_canonical_u64(inputs[i]))?;
    }
    Ok(())
}
pub fn input_targets_2d(
    pw: &mut PartialWitness<F>,
    targets: Vec<Vec<Target>>,
    inputs: Vec<Vec<u64>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let x = targets.len();
    for i in 0..x {
        input_targets_1d(pw, targets[i].clone(), inputs[i].clone())?;
    }
    Ok(())
}
pub fn input_targets_3d(
    pw: &mut PartialWitness<F>,
    targets: Vec<Vec<Vec<Target>>>,
    inputs: Vec<Vec<Vec<u64>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let x = targets.len();
    for i in 0..x {
        input_targets_2d(pw, targets[i].clone(), inputs[i].clone())?;
    }
    Ok(())
}

// 可选：如果你还想用到 std::result::Result，可以在需要处显式起别名：
// pub type StdResult<T, E> = std::result::Result<T, E>;
