use plonky2::field::goldilocks_field::GoldilocksField as F;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::HashOut;
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::plonk::config::Hasher;
use pyo3::prelude::*;

// 引入 prove.rs 模块
mod prove;

fn poseidon_single_u64(inputs: Vec<u64>) -> u64 {
    let elems: Vec<F> = inputs.into_iter().map(F::from_canonical_u64).collect();
    let out: HashOut<F> = PoseidonHash::hash_no_pad(&elems);
    out.elements[0].to_canonical_u64()
}

#[pyfunction]
fn single_hash(input: Vec<u64>) -> PyResult<u64> {
    Ok(poseidon_single_u64(input))
}

#[pyfunction]
fn step1prove(inputs: Vec<Vec<u64>>) -> PyResult<Vec<u64>> {
    let outputs = inputs.into_iter().map(poseidon_single_u64).collect();
    Ok(outputs)
}

#[pyfunction]
fn batch_hash(inputs: Vec<Vec<u64>>) -> PyResult<Vec<u64>> {
    let outputs = inputs.into_iter().map(poseidon_single_u64).collect();
    Ok(outputs)
}

#[pyfunction]
fn verify_ids_sorted_by_distance(
    centroids: Vec<Vec<u64>>,
    x: Vec<u64>,
    ids: Vec<u64>,
) -> PyResult<bool> {
    match prove::prove_ids_sorted_by_distance(centroids, x, ids) {
        Ok((_proof, _data)) => Ok(true),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "proof failed: {e}"
        ))),
    }
}

#[pymodule]
fn zk_IVF_PQ(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(single_hash, m)?)?;
    m.add_function(wrap_pyfunction!(batch_hash, m)?)?;
    m.add_function(wrap_pyfunction!(verify_ids_sorted_by_distance, m)?)?;
    Ok(())
}
