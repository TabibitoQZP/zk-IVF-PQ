pub mod brute_force;
pub mod circuit_ivf_pq;
pub mod hash_gadgets;
pub mod ivf_flat;
pub mod ivf_pq;
pub mod ivf_pq_verify;
pub mod merkle_commit;
pub mod pq_flat;
pub mod pq_flat_com;
pub mod pq_flat_verify;
pub mod prelude;
pub mod utils;

use crate::brute_force::proof::{brute_force_proof, sort_brute_force_proof};
use crate::hash_gadgets::hash_u64;
use crate::ivf_flat::proof::ivf_flat_proof;
use crate::ivf_pq::proof::ivf_pq_proof;
use crate::ivf_pq_verify::proof::ivf_pq_verify_proof;
use crate::merkle_commit::proof::{merkle_commit_plain_proof, merkle_commit_proof};
use crate::pq_flat::proof::pq_flat_proof;
use crate::pq_flat_com::proof::pq_flat_com_proof;
use crate::pq_flat_verify::proof::pq_flat_verify_proof;
use pyo3::prelude::*;
use std::error::Error;

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
fn py_merkle_commit_proof(leaves: Vec<Vec<u64>>) -> PyResult<bool> {
    let corr = merkle_commit_proof(leaves).is_ok();
    Ok(corr)
}

#[pyfunction]
fn py_merkle_commit_plain_proof(leaves: Vec<Vec<u64>>) -> PyResult<bool> {
    let corr = merkle_commit_plain_proof(leaves).is_ok();
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
fn py_sort_brute_force_proof(
    src_vecs: Vec<Vec<u64>>, // (N,D)
    query: Vec<u64>,         // (D,)
    top_k: u64,
) -> PyResult<bool> {
    let corr = sort_brute_force_proof(src_vecs, query, top_k).is_ok();
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
fn py_pq_flat_verify_proof(
    codebooks: Vec<Vec<Vec<u64>>>, // (M,K,d)
    query: Vec<u64>,               // (D,)
    pq_vecs: Vec<Vec<u64>>,        // (N,M)
    sorted_idx_dis: Vec<Vec<u64>>, // (N,2)
) -> PyResult<bool> {
    // let corr = pq_flat_verify_proof(codebooks, query, pq_vecs, sorted_idx_dis).is_ok();
    // Ok(corr)
    if let Err(e) = pq_flat_verify_proof(codebooks, query, pq_vecs, sorted_idx_dis) {
        eprintln!("error: {e}"); // Display：更简洁
        let mut src = e.source();
        while let Some(cause) = src {
            // 打印 error chain（根因）
            eprintln!("  caused by: {cause}");
            src = cause.source();
        }
        return Ok(false);
    }
    Ok(true)
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
fn py_ivf_pq_verify_proof(
    ivf_centers: Vec<Vec<i64>>,      // (n_list,D)
    query: Vec<i64>,                 // (D,)
    sorted_idx_dis: Vec<Vec<i64>>,   // (n_list,2)
    filtered_centers: Vec<Vec<i64>>, // (n_probe,D)
    probe_count: Vec<i64>,           // (n_probe,)
    filtered_vecs: Vec<Vec<i64>>,    // (max_sz,M)
    vecs_cluster_hot: Vec<Vec<i64>>, // (max_sz,n_probe)
    codebooks: Vec<Vec<Vec<i64>>>,   // (M,K,d)
) -> PyResult<bool> {
    // let corr = ivf_pq_verify_proof(
    //     ivf_centers,
    //     query,
    //     sorted_idx_dis,
    //     filtered_centers,
    //     probe_count,
    //     filtered_vecs,
    //     vecs_cluster_hot,
    //     codebooks,
    // )
    // .is_ok();
    // Ok(corr)
    if let Err(e) = ivf_pq_verify_proof(
        ivf_centers,
        query,
        sorted_idx_dis,
        filtered_centers,
        probe_count,
        filtered_vecs,
        vecs_cluster_hot,
        codebooks,
    ) {
        eprintln!("error: {e}"); // Display：更简洁
        let mut src = e.source();
        while let Some(cause) = src {
            // 打印 error chain（根因）
            eprintln!("  caused by: {cause}");
            src = cause.source();
        }
        return Ok(false);
    }
    Ok(true)
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
    m.add_function(wrap_pyfunction!(py_merkle_commit_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_merkle_commit_plain_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_brute_force_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_sort_brute_force_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_ivf_flat_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_pq_flat_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_pq_flat_verify_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_pq_flat_com_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_ivf_pq_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_ivf_pq_verify_proof, m)?)?;
    Ok(())
}
