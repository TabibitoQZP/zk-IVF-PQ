// 有pub会给外部crate用, 无pub是给内部crate用
// pub mod common_gadgets;
pub mod hash_gadgets;
pub mod ivf_flat;
// pub mod ivfpq;
// pub mod nn_gadgets;
pub mod prelude;
pub mod utils;

use crate::hash_gadgets::hash_u64;
use crate::ivf_flat::proof::ivf_flat_proof;
// use crate::nn_gadgets::nn_prove;
use pyo3::prelude::*;

#[pyfunction]
fn single_hash(input: Vec<u64>) -> PyResult<u64> {
    Ok(hash_u64(input))
}

// #[pyfunction]
// fn py_nn_prove(
//     src_vecs: Vec<Vec<u64>>,
//     query: Vec<u64>,
//     root: u64,
//     sorted_idx_dis: Vec<Vec<u64>>,
// ) -> PyResult<bool> {
//     let corr = nn_prove(src_vecs, query, root, sorted_idx_dis).is_ok();
//     Ok(corr)
// }

#[pyfunction]
fn py_ivf_flat_proof(
    ivf_centers: Vec<Vec<u64>>,      // (n_list,d)
    query: Vec<u64>,                 // (d,)
    sorted_idx_dis: Vec<Vec<u64>>,   // (n_list,2)
    probe_count: Vec<u64>,           // (n_probe,)
    filtered_vecs: Vec<Vec<u64>>,    // (max_sz,d)
    vecs_cluster_hot: Vec<Vec<u64>>, // (max_sz,n_probe)
) -> PyResult<bool> {
    let corr = ivf_flat_proof(
        ivf_centers,
        query,
        sorted_idx_dis,
        probe_count,
        filtered_vecs,
        vecs_cluster_hot,
    )
    .is_ok();
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
    // m.add_function(wrap_pyfunction!(py_nn_prove, m)?)?;
    m.add_function(wrap_pyfunction!(py_ivf_flat_proof, m)?)?;
    Ok(())
}
