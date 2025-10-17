// 有pub会给外部crate用, 无pub是给内部crate用
pub mod hash_gadgets;
pub mod nn_gadgets;
pub mod prelude;
pub mod prove;

use crate::hash_gadgets::hash_u64;
use crate::nn_gadgets::nn_prove;
use pyo3::prelude::*;

#[pyfunction]
fn single_hash(input: Vec<u64>) -> PyResult<u64> {
    Ok(hash_u64(input))
}

#[pyfunction]
fn py_nn_prove(
    src_vecs: Vec<Vec<u64>>,
    query: Vec<u64>,
    root: u64,
    sorted_idx_dis: Vec<Vec<u64>>,
) -> PyResult<bool> {
    let corr = nn_prove(src_vecs, query, root, sorted_idx_dis).is_ok();
    Ok(corr)
}

#[pyfunction]
fn batch_hash(inputs: Vec<Vec<u64>>) -> PyResult<Vec<u64>> {
    let outputs = inputs.into_iter().map(hash_u64).collect();
    Ok(outputs)
}

#[pymodule]
fn zk_IVF_PQ(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(single_hash, m)?)?;
    m.add_function(wrap_pyfunction!(batch_hash, m)?)?;
    m.add_function(wrap_pyfunction!(py_nn_prove, m)?)?;
    Ok(())
}
