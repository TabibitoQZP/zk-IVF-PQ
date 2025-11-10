use crate::halo2_brute_force::circuit::Halo2BruteForceCircuit;
use crate::hash_gadgets::fs_oracle;
use halo2_proofs::halo2curves::bn256::{Bn256, Fr};
use halo2_proofs::plonk::{create_proof, keygen_pk, keygen_vk, verify_proof};
use halo2_proofs::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use halo2_proofs::poly::kzg::multiopen::{ProverGWC, VerifierGWC};
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::transcript::{Blake2bRead, Blake2bWrite, Challenge255};
use rand::rngs::OsRng;
use std::time::Instant;

fn estimate_k(row_count: usize) -> u32 {
    let mut size = 1usize;
    let mut k = 0u32;
    while size < row_count {
        size <<= 1;
        k += 1;
    }
    k.max(9)
}

pub fn halo2_brute_force_proof(
    src_vecs: Vec<Vec<u64>>,       // (N,D)
    query: Vec<u64>,               // (D,)
    sorted_idx_dis: Vec<Vec<u64>>, // (N,2)
) -> Result<(), Box<dyn std::error::Error>> {
    let n = src_vecs.len();
    let d = if query.is_empty() { 0 } else { query.len() };

    if n == 0 || d == 0 {
        return Err("输入数据不能为空".into());
    }

    let fs_hash = fs_oracle(query.clone(), 2);
    let fs_r = Fr::from(fs_hash[0]);
    let fs_t = Fr::from(fs_hash[1]);

    let query_fr: Vec<Fr> = query.iter().map(|&v| Fr::from(v)).collect();
    let src_fr: Vec<Vec<Fr>> = src_vecs
        .iter()
        .map(|row| row.iter().map(|&v| Fr::from(v)).collect())
        .collect();
    let sorted_pairs_u64: Vec<(u64, u64)> =
        sorted_idx_dis.iter().map(|row| (row[0], row[1])).collect();
    let sorted_pairs_fr: Vec<(Fr, Fr)> = sorted_pairs_u64
        .iter()
        .map(|&(idx, dis)| (Fr::from(idx), Fr::from(dis)))
        .collect();

    let circuit = Halo2BruteForceCircuit {
        query: query_fr.clone(),
        src_vecs: src_fr,
        sorted_idx_dis: sorted_pairs_fr,
        sorted_pairs_u64: sorted_pairs_u64.clone(),
        fs_r,
        fs_t,
    };

    let mut row_estimate = query_fr.len();
    row_estimate += n * d;
    if n > 0 {
        row_estimate += (n - 1) * 33;
    }
    row_estimate += n;
    row_estimate += sorted_idx_dis.len();
    row_estimate += 64; // buffer for selectors and copy constraints

    let k = estimate_k(row_estimate);

    let mut timer = Instant::now();
    let params = ParamsKZG::<Bn256>::setup(k, OsRng);
    println!("Halo2参数生成: {:?}", timer.elapsed());

    timer = Instant::now();
    let vk = keygen_vk(&params, &circuit)?;
    let pk = keygen_pk(&params, vk.clone(), &circuit)?;
    println!("密钥生成耗时: {:?}", timer.elapsed());

    let mut instances = vec![query_fr.clone()];
    let instance_slices: Vec<&[Fr]> = instances.iter().map(|v| v.as_slice()).collect();

    timer = Instant::now();
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    let mut rng = OsRng;
    create_proof::<KZGCommitmentScheme<Bn256>, ProverGWC<_>, _, _, _>(
        &params,
        &pk,
        &[circuit.clone()],
        &[instance_slices.as_slice()],
        &mut rng,
        &mut transcript,
    )?;
    let proof = transcript.finalize();
    println!("证明生成耗时: {:?}", timer.elapsed());

    let params_verifier = params.verifier();
    let mut verifier_transcript = Blake2bRead::<_, _, Challenge255<_>>::init(proof.as_slice());
    verify_proof::<KZGCommitmentScheme<Bn256>, VerifierGWC<_>, _, _>(
        &params_verifier,
        pk.get_vk(),
        SingleStrategy::new(&params_verifier),
        &[instance_slices.as_slice()],
        &mut verifier_transcript,
    )?;
    println!("证明验证完成");

    Ok(())
}
