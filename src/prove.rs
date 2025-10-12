use anyhow::{ensure, Result};
use plonky2::field::goldilocks_field::GoldilocksField as F;
use plonky2::field::types::Field;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::config::PoseidonGoldilocksConfig as C;
use plonky2::plonk::proof;

const DCFG: usize = 2;

fn set_vec_targets(pw: &mut PartialWitness<F>, ts: &[Target], vals: &[u64]) {
    assert_eq!(ts.len(), vals.len());
    for (t, &v) in ts.iter().zip(vals) {
        pw.set_target(*t, F::from_canonical_u64(v));
    }
}

pub fn prove_ids_sorted_by_distance(
    centroids: Vec<Vec<u64>>, // (n_list, D)
    x: Vec<u64>,              // (D,)
    ids: Vec<u64>,            // (n_list,)
) -> Result<(
    proof::ProofWithPublicInputs<F, C, DCFG>,
    CircuitData<F, C, DCFG>,
)> {
    let n_list = centroids.len();
    ensure!(n_list > 1, "n_list must be > 1");
    let d = x.len();
    ensure!(
        centroids.iter().all(|row| row.len() == d),
        "centroids shape mismatch"
    );
    ensure!(ids.len() == n_list, "ids length mismatch");

    // ---- build circuit ----
    let cfg = CircuitConfig::standard_recursion_zk_config();
    let mut builder = CircuitBuilder::<F, DCFG>::new(cfg);

    // virtual targets
    let x_t: Vec<Target> = (0..d).map(|_| builder.add_virtual_target()).collect();
    let c_t: Vec<Vec<Target>> = (0..n_list)
        .map(|_| (0..d).map(|_| builder.add_virtual_target()).collect())
        .collect();
    let ids_t: Vec<Target> = (0..n_list).map(|_| builder.add_virtual_target()).collect();

    // range checks: coords 16-bit; ids: enough bits to cover 0..n_list-1
    for &t in &x_t {
        builder.range_check(t, 16);
    }
    for row in &c_t {
        for &t in row {
            builder.range_check(t, 16);
        }
    }
    let id_bits = (64 - (n_list as u64).next_power_of_two().leading_zeros()) as usize;
    for &t in &ids_t {
        builder.range_check(t, id_bits.max(1));
    }

    // distances
    let mut dists: Vec<Target> = Vec::with_capacity(n_list);
    for i in 0..n_list {
        let mut acc = builder.zero();
        for j in 0..d {
            let diff = builder.sub(c_t[i][j], x_t[j]);
            let sq = builder.mul(diff, diff);
            acc = builder.add(acc, sq);
        }
        builder.range_check(acc, 64);
        dists.push(acc);
    }

    // 建电路后、进入行列和约束前，先缓存常量 1 的 Target
    let one = builder.one();

    // permutation matrix eq[k][i]
    let mut eq: Vec<Vec<BoolTarget>> = vec![Vec::with_capacity(n_list); n_list];

    // row sum == 1
    for k in 0..n_list {
        let mut row_sum = builder.zero();
        for i in 0..n_list {
            let ci = builder.constant(F::from_canonical_u64(i as u64));
            let b = builder.is_equal(ids_t[k], ci);
            builder.assert_bool(b);
            row_sum = builder.add(row_sum, b.target);
            eq[k].push(b);
        }
        builder.connect(row_sum, one);
    }

    // column sum == 1
    for i in 0..n_list {
        let mut col_sum = builder.zero();
        for k in 0..n_list {
            col_sum = builder.add(col_sum, eq[k][i].target);
        }
        builder.connect(col_sum, one);
    }

    // sorted distances by ids
    let mut sorted: Vec<Target> = Vec::with_capacity(n_list);
    for k in 0..n_list {
        let mut acc = builder.zero();
        for i in 0..n_list {
            let term = builder.mul(dists[i], eq[k][i].target);
            acc = builder.add(acc, term);
        }
        sorted.push(acc);
    }

    // non-decreasing
    for k in 0..(n_list - 1) {
        let delta = builder.sub(sorted[k + 1], sorted[k]);
        builder.range_check(delta, 64);
    }

    // prove & verify
    let mut pw = PartialWitness::<F>::new();
    set_vec_targets(&mut pw, &x_t, &x);
    for (row_t, row_v) in c_t.iter().zip(centroids.iter()) {
        set_vec_targets(&mut pw, row_t, row_v);
    }
    set_vec_targets(&mut pw, &ids_t, &ids);

    let data = builder.build::<C>();
    let prf = data.prove(pw)?;
    data.verify(prf.clone())?;

    Ok((prf, data))
}
