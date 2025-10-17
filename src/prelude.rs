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
    CircuitBuilder::<F, D>::new(cfg)
}

// 可选：如果你还想用到 std::result::Result，可以在需要处显式起别名：
// pub type StdResult<T, E> = std::result::Result<T, E>;
