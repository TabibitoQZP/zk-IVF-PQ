pub mod brute_force;
pub mod halo2_brute_force;
pub mod hash_gadgets;
pub mod ivf_flat;
pub mod ivf_pq;
pub mod pq_flat;
pub mod pq_flat_acc;
pub mod pq_flat_accel;
pub mod pq_flat_com;
pub mod prelude;
pub mod utils;

use crate::brute_force::proof::brute_force_proof;
use crate::halo2_brute_force::proof::halo2_brute_force_proof;
use crate::hash_gadgets::hash_u64;
use crate::ivf_flat::proof::ivf_flat_proof;
use crate::ivf_pq::proof::ivf_pq_proof;
use crate::pq_flat::proof::pq_flat_proof;
use crate::pq_flat_acc::proof::pq_flat_acc_proof;
use crate::pq_flat_accel::proof::pq_flat_accel_proof;
use crate::pq_flat_com::proof::pq_flat_com_proof;
use pyo3::prelude::*;

#[pyfunction]
fn single_hash(input: Vec<u64>) -> PyResult<u64> {
    Ok(hash_u64(input))
}

#[pyfunction]
fn py_pq_flat_com_proof(
    codebooks: Vec<Vec<Vec<u64>>>, // (M,K,d)
    query: Vec<u64>,               // (D,)
    pq_vecs: Vec<Vec<u64>>,        // (N,M)
    sorted_idx_dis: Vec<Vec<u64>>, // (N,2)
) -> PyResult<bool> {
    let corr = pq_flat_com_proof(codebooks, query, pq_vecs, sorted_idx_dis).is_ok();
    Ok(corr)
}

#[pyfunction]
fn py_brute_force_proof(
    src_vecs: Vec<Vec<u64>>,       // (N,D)
    query: Vec<u64>,               // (D,)
    sorted_idx_dis: Vec<Vec<u64>>, // (N,2)
) -> PyResult<bool> {
    let corr = brute_force_proof(src_vecs, query, sorted_idx_dis).is_ok();
    Ok(corr)
}

#[pyfunction]
fn py_halo2_brute_force_proof(
    src_vecs: Vec<Vec<u64>>,       // (N,D)
    query: Vec<u64>,               // (D,)
    sorted_idx_dis: Vec<Vec<u64>>, // (N,2)
) -> PyResult<bool> {
    let corr = halo2_brute_force_proof(src_vecs, query, sorted_idx_dis).is_ok();
    Ok(corr)
}

#[pyfunction]
fn py_pq_flat_proof(
    codebooks: Vec<Vec<Vec<u64>>>, // (M,K,d)
    query: Vec<u64>,               // (D,)
    pq_vecs: Vec<Vec<u64>>,        // (N,M)
    sorted_idx_dis: Vec<Vec<u64>>, // (N,2)
) -> PyResult<bool> {
    let corr = pq_flat_proof(codebooks, query, pq_vecs, sorted_idx_dis).is_ok();
    Ok(corr)
}

#[pyfunction]
fn py_pq_flat_accel_proof(
    codebooks: Vec<Vec<Vec<u64>>>, // (M,K,d)
    query: Vec<u64>,               // (D,)
    pq_vecs: Vec<Vec<u64>>,        // (N,M)
    sorted_idx_dis: Vec<Vec<u64>>, // (N,2)
) -> PyResult<bool> {
    let corr = pq_flat_accel_proof(codebooks, query, pq_vecs, sorted_idx_dis).is_ok();
    Ok(corr)
}

#[pyfunction]
fn py_pq_flat_acc_proof(
    codebooks: Vec<Vec<Vec<u64>>>,       // (M,K,d)
    query: Vec<u64>,                     // (D,)
    pq_vecs: Vec<Vec<u64>>,              // (N,M)
    pq_sub_distances: Vec<Vec<u64>>,     // (N,M)
    unused_table_entries: Vec<Vec<u64>>, // (M*K*N - N*M, 4)
    sorted_idx_dis: Vec<Vec<u64>>,       // (N,2)
) -> PyResult<bool> {
    let corr = pq_flat_acc_proof(
        codebooks,
        query,
        pq_vecs,
        pq_sub_distances,
        unused_table_entries,
        sorted_idx_dis,
    )
    .is_ok();
    Ok(corr)
}

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
fn py_ivf_pq_proof(
    ivf_centers: Vec<Vec<u64>>,      // (n_list,D)
    query: Vec<u64>,                 // (D,)
    sorted_idx_dis: Vec<Vec<u64>>,   // (n_list,2)
    filtered_centers: Vec<Vec<u64>>, // (n_probe,D)
    probe_count: Vec<u64>,           // (n_probe,)
    filtered_vecs: Vec<Vec<u64>>,    // (max_sz,M)
    vecs_cluster_hot: Vec<Vec<u64>>, // (max_sz,n_probe)
    codebooks: Vec<Vec<Vec<u64>>>,   // (M,K,d)
) -> PyResult<bool> {
    let corr = ivf_pq_proof(
        ivf_centers,
        query,
        sorted_idx_dis,
        filtered_centers,
        probe_count,
        filtered_vecs,
        vecs_cluster_hot,
        codebooks,
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
    // 暴露的哈希函数, 用于计算root
    m.add_function(wrap_pyfunction!(single_hash, m)?)?;
    m.add_function(wrap_pyfunction!(batch_hash, m)?)?;

    // 各种向量数据库的证明系统
    m.add_function(wrap_pyfunction!(py_brute_force_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_halo2_brute_force_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_ivf_flat_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_pq_flat_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_pq_flat_accel_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_pq_flat_acc_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_pq_flat_com_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_ivf_pq_proof, m)?)?;
    Ok(())
}
