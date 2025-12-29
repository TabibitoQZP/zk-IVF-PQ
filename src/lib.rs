pub mod brute_force;
pub mod circuit_ivf_pq;
pub mod commit_eval;
pub mod hash_gadgets;
pub mod ivf_flat;
pub mod ivf_flat_verify;
pub mod ivf_pq;
pub mod ivf_pq_verify;
pub mod merkle_commit;
pub mod merkle_ver;
pub mod pq_flat;
pub mod pq_flat_com;
pub mod pq_flat_verify;
pub mod prelude;
pub mod utils;

use crate::brute_force::proof::{brute_force_proof, sort_brute_force_proof};
use crate::circuit_ivf_pq::proof::circuit_ivf_pq_proof;
use crate::commit_eval::fri::run;
use crate::commit_eval::merkle::merkle_run;
use crate::hash_gadgets::hash_u64;
use crate::ivf_flat::proof::ivf_flat_proof;
use crate::ivf_flat_verify::proof::ivf_flat_verify_proof;
use crate::ivf_pq::proof::ivf_pq_proof;
use crate::ivf_pq_verify::proof::ivf_pq_verify_proof;
use crate::merkle_commit::proof::{merkle_commit_plain_proof, merkle_commit_proof};
use crate::merkle_ver::circuit_based_proof::circuit_based_ivf_pq_proof;
use crate::merkle_ver::set_based_proof::{set_based_gate, set_based_ivf_pq_proof};
use crate::merkle_ver::standalone_commitment::standalone_commitment_proof;
use crate::pq_flat::proof::pq_flat_proof;
use crate::pq_flat_com::proof::pq_flat_com_proof;
use crate::pq_flat_verify::proof::pq_flat_verify_proof;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::error::Error;

#[pyfunction]
pub fn py_set_based_gate(
    M: usize,
    K: usize,
    d: usize,
    n_list: usize,
    n_probe: usize,
    n: usize,
    top_k: usize,
    merkled: bool,
) -> PyResult<usize> {
    let gates = set_based_gate(M, K, d, n_list, n_probe, n, top_k, merkled);
    Ok(gates)
}

#[pyfunction]
pub fn py_merkle(n_list: usize, n: usize, M: usize, loop_time: usize) -> PyResult<(f64, u64)> {
    let (duration, memory_peak) = merkle_run(n_list, n, M, loop_time);
    Ok((duration, memory_peak))
}

#[pyfunction]
pub fn py_fri(max_exponent: usize, poseidon: bool) -> PyResult<(f64, u64)> {
    let (duration, memory_peak) = run(max_exponent, poseidon);
    Ok((duration, memory_peak))
}

#[pyfunction]
pub fn py_standalone_commitment(
    query: Vec<i64>,               // 查询向量 (D,)
    ivf_center: Vec<Vec<i64>>,     // ivf簇中心 (n_list,D)
    cluster_idxes: Vec<i64>,       // 簇索引 (n_probe,)
    vpqss: Vec<Vec<Vec<i64>>>,     // 这里给原始向量, 手动改one-hot (n_probe,n,M)
    valids: Vec<Vec<i64>>,         // vpqss中向量是否valid (n_probe,n)
    itemss: Vec<Vec<i64>>,         // vpqss中向量对应的查询量 (n_probe,n)
    codebooks: Vec<Vec<Vec<i64>>>, // 全局码本 (M,K,d)
    ivf_roots: Vec<u64>,           // 这里给一下ivf各个root, 用来手算和还原数据 (n_list,)
) -> PyResult<(f64, f64, f64, u64, u64, u64)> {
    let (build_time, prove_time, verify_time, proof_size, memory_used, num_gates) =
        standalone_commitment_proof(
            query,         // 查询向量 (D,)
            ivf_center,    // ivf簇中心 (n_list,D)
            cluster_idxes, // 簇索引 (n_probe,)
            vpqss,         // 这里给原始向量, 手动改one-hot (n_probe,n,M)
            valids,        // vpqss中向量是否valid (n_probe,n)
            itemss,        // vpqss中向量对应的查询量 (n_probe,n)
            codebooks,     // 全局码本 (M,K,d)
            ivf_roots,     // 这里给一下ivf各个root, 用来手算和还原数据 (n_list,)
        )
        .map_err(|e| PyRuntimeError::new_err(format!("circuit_ivf_pq_proof failed: {e}")))?;

    Ok((
        build_time,
        prove_time,
        verify_time,
        proof_size,
        memory_used,
        num_gates,
    ))
}

#[pyfunction]
pub fn py_set_based_with_merkle(
    query: Vec<i64>,               // 查询向量 (D,)
    ivf_center: Vec<Vec<i64>>,     // ivf簇中心 (n_list,D)
    vpqss: Vec<Vec<Vec<i64>>>,     // 这里给原始向量, 手动改one-hot (n_probe,n,M)
    valids: Vec<Vec<i64>>,         // vpqss中向量是否valid (n_probe,n)
    itemss: Vec<Vec<i64>>,         // vpqss中向量对应的查询量 (n_probe,n)
    codebooks: Vec<Vec<Vec<i64>>>, // 全局码本 (M,K,d)
    ivf_roots: Vec<u64>,           // 这里给一下ivf各个root, 用来手算和还原数据 (n_list,)
    top_k: i64,                    // 明确取哪top_k
    // 后面的可以在rust内部算, 也可以python端算完传入, 这里用传入实现, 懒得写了...
    cluster_idx_dis: Vec<Vec<i64>>,        // (n_list,2)
    ordered_vpqss_item_dis: Vec<Vec<i64>>, // vpqss中计算的距离和item集合 (n_probe*n,2)
) -> PyResult<(f64, f64, f64, u64, u64, u64)> {
    let (build_time, prove_time, verify_time, proof_size, memory_used, num_gates) =
        set_based_ivf_pq_proof(
            query,                  // 查询向量 (D,)
            ivf_center,             // ivf簇中心 (n_list,D)
            vpqss,                  // 这里给原始向量, 手动改one-hot (n_probe,n,M)
            valids,                 // vpqss中向量是否valid (n_probe,n)
            itemss,                 // vpqss中向量对应的查询量 (n_probe,n)
            codebooks,              // 全局码本 (M,K,d)
            ivf_roots,              // 这里给一下ivf各个root, 用来手算和还原数据 (n_list,)
            top_k,                  // 明确取哪top_k
            cluster_idx_dis,        // (n_list,2)
            ordered_vpqss_item_dis, // vpqss中计算的距离和item集合 (n_probe*n,2)
            true,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("circuit_ivf_pq_proof failed: {e}")))?;

    Ok((
        build_time,
        prove_time,
        verify_time,
        proof_size,
        memory_used,
        num_gates,
    ))
}

#[pyfunction]
pub fn py_set_based_without_merkle(
    query: Vec<i64>,               // 查询向量 (D,)
    ivf_center: Vec<Vec<i64>>,     // ivf簇中心 (n_list,D)
    vpqss: Vec<Vec<Vec<i64>>>,     // 这里给原始向量, 手动改one-hot (n_probe,n,M)
    valids: Vec<Vec<i64>>,         // vpqss中向量是否valid (n_probe,n)
    itemss: Vec<Vec<i64>>,         // vpqss中向量对应的查询量 (n_probe,n)
    codebooks: Vec<Vec<Vec<i64>>>, // 全局码本 (M,K,d)
    ivf_roots: Vec<u64>,           // 这里给一下ivf各个root, 用来手算和还原数据 (n_list,)
    top_k: i64,                    // 明确取哪top_k
    // 后面的可以在rust内部算, 也可以python端算完传入, 这里用传入实现, 懒得写了...
    cluster_idx_dis: Vec<Vec<i64>>,        // (n_list,2)
    ordered_vpqss_item_dis: Vec<Vec<i64>>, // vpqss中计算的距离和item集合 (n_probe*n,2)
) -> PyResult<(f64, f64, f64, u64, u64, u64)> {
    let (build_time, prove_time, verify_time, proof_size, memory_used, num_gates) =
        set_based_ivf_pq_proof(
            query,                  // 查询向量 (D,)
            ivf_center,             // ivf簇中心 (n_list,D)
            vpqss,                  // 这里给原始向量, 手动改one-hot (n_probe,n,M)
            valids,                 // vpqss中向量是否valid (n_probe,n)
            itemss,                 // vpqss中向量对应的查询量 (n_probe,n)
            codebooks,              // 全局码本 (M,K,d)
            ivf_roots,              // 这里给一下ivf各个root, 用来手算和还原数据 (n_list,)
            top_k,                  // 明确取哪top_k
            cluster_idx_dis,        // (n_list,2)
            ordered_vpqss_item_dis, // vpqss中计算的距离和item集合 (n_probe*n,2)
            false,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("circuit_ivf_pq_proof failed: {e}")))?;

    Ok((
        build_time,
        prove_time,
        verify_time,
        proof_size,
        memory_used,
        num_gates,
    ))
}

#[pyfunction]
pub fn py_circuit_based_with_merkle(
    query: Vec<i64>,               // 查询向量 (D,)
    ivf_center: Vec<Vec<i64>>,     // ivf簇中心 (n_list,D)
    cluster_idxes: Vec<i64>,       // 簇索引 (n_probe,)
    vpqss: Vec<Vec<Vec<i64>>>,     // 这里给原始向量, 手动改one-hot (n_probe,n,M)
    valids: Vec<Vec<i64>>,         // vpqss中向量是否valid (n_probe,n)
    itemss: Vec<Vec<i64>>,         // vpqss中向量对应的查询量 (n_probe,n)
    codebooks: Vec<Vec<Vec<i64>>>, // 全局码本 (M,K,d)
    ivf_roots: Vec<u64>,           // 这里给一下ivf各个root, 用来手算和还原数据 (n_list,)
    top_k: i64,                    // 明确取哪top_k
) -> PyResult<(f64, f64, f64, u64, u64, u64)> {
    let (build_time, prove_time, verify_time, proof_size, memory_used, num_gates) =
        circuit_based_ivf_pq_proof(
            query,         // 查询向量 (D,)
            ivf_center,    // ivf簇中心 (n_list,D)
            cluster_idxes, // 簇索引 (n_probe,)
            vpqss,         // 这里给原始向量, 手动改one-hot (n_probe,n,M)
            valids,        // vpqss中向量是否valid (n_probe,n)
            itemss,        // vpqss中向量对应的查询量 (n_probe,n)
            codebooks,     // 全局码本 (M,K,d)
            ivf_roots,     // 这里给一下ivf各个root, 用来手算和还原数据 (n_list,)
            top_k,         // 明确取哪top_k
            true,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("circuit_ivf_pq_proof failed: {e}")))?;

    Ok((
        build_time,
        prove_time,
        verify_time,
        proof_size,
        memory_used,
        num_gates,
    ))
}

#[pyfunction]
pub fn py_circuit_based_without_merkle(
    query: Vec<i64>,               // 查询向量 (D,)
    ivf_center: Vec<Vec<i64>>,     // ivf簇中心 (n_list,D)
    cluster_idxes: Vec<i64>,       // 簇索引 (n_probe,)
    vpqss: Vec<Vec<Vec<i64>>>,     // 这里给原始向量, 手动改one-hot (n_probe,n,M)
    valids: Vec<Vec<i64>>,         // vpqss中向量是否valid (n_probe,n)
    itemss: Vec<Vec<i64>>,         // vpqss中向量对应的查询量 (n_probe,n)
    codebooks: Vec<Vec<Vec<i64>>>, // 全局码本 (M,K,d)
    ivf_roots: Vec<u64>,           // 这里给一下ivf各个root, 用来手算和还原数据 (n_list,)
    top_k: i64,                    // 明确取哪top_k
) -> PyResult<(f64, f64, f64, u64, u64, u64)> {
    let (build_time, prove_time, verify_time, proof_size, memory_used, num_gates) =
        circuit_based_ivf_pq_proof(
            query,         // 查询向量 (D,)
            ivf_center,    // ivf簇中心 (n_list,D)
            cluster_idxes, // 簇索引 (n_probe,)
            vpqss,         // 这里给原始向量, 手动改one-hot (n_probe,n,M)
            valids,        // vpqss中向量是否valid (n_probe,n)
            itemss,        // vpqss中向量对应的查询量 (n_probe,n)
            codebooks,     // 全局码本 (M,K,d)
            ivf_roots,     // 这里给一下ivf各个root, 用来手算和还原数据 (n_list,)
            top_k,         // 明确取哪top_k
            false,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("circuit_ivf_pq_proof failed: {e}")))?;

    Ok((
        build_time,
        prove_time,
        verify_time,
        proof_size,
        memory_used,
        num_gates,
    ))
}

#[pyfunction]
fn py_circuit_ivf_pq_proof(
    query: Vec<i64>,               // 查询向量 (D,)
    ivf_centers: Vec<Vec<i64>>,    // ivf簇中心 *(n_list,D)
    vecs: Vec<Vec<Vec<Vec<i64>>>>, // 这里每个都固定给到 (n_probe,max_sz,M,K)
    hot: Vec<Vec<i64>>,            // 针对vecs是否valid
    codebooks: Vec<Vec<Vec<i64>>>, // 全局码本 (M,K,d)
    top_k: i64,                    // 明确取哪top_k
) -> PyResult<(f64, f64, f64, u64, u64, u64)> {
    let (build_time, prove_time, verify_time, proof_size, memory_used, num_gates) =
        circuit_ivf_pq_proof(query, ivf_centers, vecs, hot, codebooks, top_k)
            .map_err(|e| PyRuntimeError::new_err(format!("circuit_ivf_pq_proof failed: {e}")))?;

    Ok((
        build_time,
        prove_time,
        verify_time,
        proof_size,
        memory_used,
        num_gates,
    ))
}

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
fn py_ivf_flat_verify_proof(
    ivf_centers: Vec<Vec<u64>>,    // (n_list,d)
    query: Vec<u64>,               // (d,)
    sorted_idx_dis: Vec<Vec<u64>>, // (n_list,2)
    vecss: Vec<Vec<Vec<u64>>>,     // (n_probe,n,d)
    valids: Vec<Vec<u64>>,         // (n_probe,n)
    itemss: Vec<Vec<u64>>,         // (n_probe,n)
    top_k: usize,                  // 明确取哪top_k
) -> PyResult<bool> {
    let corr = ivf_flat_verify_proof(
        ivf_centers,
        query,
        sorted_idx_dis,
        vecss,
        valids,
        itemss,
        top_k,
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

    // commitment对比
    m.add_function(wrap_pyfunction!(py_fri, m)?)?;
    m.add_function(wrap_pyfunction!(py_merkle, m)?)?;

    // 带merkle的
    m.add_function(wrap_pyfunction!(py_standalone_commitment, m)?)?;
    m.add_function(wrap_pyfunction!(py_circuit_based_with_merkle, m)?)?;
    m.add_function(wrap_pyfunction!(py_circuit_based_without_merkle, m)?)?;
    m.add_function(wrap_pyfunction!(py_set_based_with_merkle, m)?)?;
    m.add_function(wrap_pyfunction!(py_set_based_without_merkle, m)?)?;
    m.add_function(wrap_pyfunction!(py_set_based_gate, m)?)?;

    // 各种向量数据库的证明系统
    m.add_function(wrap_pyfunction!(py_merkle_commit_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_merkle_commit_plain_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_brute_force_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_sort_brute_force_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_ivf_flat_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_ivf_flat_verify_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_pq_flat_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_pq_flat_verify_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_pq_flat_com_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_ivf_pq_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_circuit_ivf_pq_proof, m)?)?;
    m.add_function(wrap_pyfunction!(py_ivf_pq_verify_proof, m)?)?;
    Ok(())
}
