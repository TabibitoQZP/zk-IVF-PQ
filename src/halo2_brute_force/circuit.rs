use halo2_proofs::circuit::{AssignedCell, Layouter, Region, SimpleFloorPlanner, Value};
use halo2_proofs::halo2curves::bn256::Fr;
use halo2_proofs::plonk::{Circuit, ConstraintSystem, Error, Selector};
use halo2_proofs::poly::Rotation;

#[derive(Clone, Debug)]
pub struct RangeCheckConfig {
    acc: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,
    bit: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,
    q_step: Selector,
    q_last: Selector,
}

#[derive(Clone, Debug)]
pub struct BruteForceConfig {
    pub src: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,
    pub query: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,
    pub diff: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,
    pub sq: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,
    pub acc_prev: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,
    pub acc: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,
    pub param_t: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,
    pub param_r: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,
    pub q_calc: Selector,
    pub q_prod: Selector,
    pub range: RangeCheckConfig,
    pub instance: halo2_proofs::plonk::Column<halo2_proofs::plonk::Instance>,
}

#[derive(Clone, Debug)]
pub struct RangeCheckChip {
    config: RangeCheckConfig,
}

impl RangeCheckChip {
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        acc: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,
        bit: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,
    ) -> RangeCheckConfig {
        let q_step = meta.complex_selector();
        let q_last = meta.selector();

        meta.create_gate("range step", |meta| {
            let q = meta.query_selector(q_step);
            let acc_prev = meta.query_advice(acc, Rotation::prev());
            let acc_cur = meta.query_advice(acc, Rotation::cur());
            let bit_val = meta.query_advice(bit, Rotation::cur());
            let two = halo2_proofs::plonk::Expression::Constant(Fr::from(2u64));
            vec![
                q.clone() * (acc_prev - (acc_cur.clone() * two + bit_val.clone())),
                q * bit_val.clone()
                    * (bit_val - halo2_proofs::plonk::Expression::Constant(Fr::one())),
            ]
        });

        meta.create_gate("range last", |meta| {
            let q = meta.query_selector(q_last);
            let acc_cur = meta.query_advice(acc, Rotation::cur());
            vec![q * acc_cur]
        });

        RangeCheckConfig {
            acc,
            bit,
            q_step,
            q_last,
        }
    }

    pub fn construct(config: RangeCheckConfig) -> Self {
        Self { config }
    }

    pub fn assign(&self, layouter: &mut impl Layouter<Fr>, value: u64) -> Result<(), Error> {
        let config = self.config.clone();
        layouter.assign_region(
            || "range check",
            |mut region: Region<'_, Fr>| {
                let mut offset = 0;
                let mut current = value;

                region.assign_advice(
                    || "acc",
                    config.acc,
                    offset,
                    || Value::known(Fr::from(current)),
                )?;
                region.assign_advice(|| "bit", config.bit, offset, || Value::known(Fr::zero()))?;
                offset += 1;

                for _ in 0..32 {
                    let bit_val = current & 1;
                    current >>= 1;
                    region.assign_advice(
                        || "acc",
                        config.acc,
                        offset,
                        || Value::known(Fr::from(current)),
                    )?;
                    region.assign_advice(
                        || "bit",
                        config.bit,
                        offset,
                        || Value::known(Fr::from(bit_val as u64)),
                    )?;
                    config.q_step.enable(&mut region, offset)?;
                    offset += 1;
                }

                config.q_last.enable(&mut region, offset - 1)?;
                Ok(())
            },
        )
    }
}

#[derive(Clone, Debug)]
pub struct Halo2BruteForceCircuit {
    pub query: Vec<Fr>,
    pub src_vecs: Vec<Vec<Fr>>,
    pub sorted_idx_dis: Vec<(Fr, Fr)>,
    pub sorted_pairs_u64: Vec<(u64, u64)>,
    pub fs_r: Fr,
    pub fs_t: Fr,
}

impl Circuit<Fr> for Halo2BruteForceCircuit {
    type Config = BruteForceConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        let query_len = self.query.len();
        let n = self.src_vecs.len();
        let d = if query_len == 0 {
            0
        } else {
            self.src_vecs[0].len()
        };
        Self {
            query: vec![Fr::zero(); query_len],
            src_vecs: vec![vec![Fr::zero(); d]; n],
            sorted_idx_dis: vec![(Fr::zero(), Fr::zero()); self.sorted_idx_dis.len()],
            sorted_pairs_u64: vec![(0u64, 0u64); self.sorted_pairs_u64.len()],
            fs_r: Fr::zero(),
            fs_t: Fr::zero(),
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let src = meta.advice_column();
        let query = meta.advice_column();
        let diff = meta.advice_column();
        let sq = meta.advice_column();
        let acc_prev = meta.advice_column();
        let acc = meta.advice_column();
        let param_t = meta.advice_column();
        let param_r = meta.advice_column();

        meta.enable_equality(src);
        meta.enable_equality(query);
        meta.enable_equality(diff);
        meta.enable_equality(sq);
        meta.enable_equality(acc_prev);
        meta.enable_equality(acc);
        meta.enable_equality(param_t);
        meta.enable_equality(param_r);

        let instance = meta.instance_column();
        meta.enable_equality(instance);

        let q_calc = meta.selector();
        meta.create_gate("distance calc", |meta| {
            let q = meta.query_selector(q_calc);
            let src_val = meta.query_advice(src, Rotation::cur());
            let query_val = meta.query_advice(query, Rotation::cur());
            let diff_val = meta.query_advice(diff, Rotation::cur());
            let sq_val = meta.query_advice(sq, Rotation::cur());
            let acc_prev_val = meta.query_advice(acc_prev, Rotation::cur());
            let acc_val = meta.query_advice(acc, Rotation::cur());
            vec![
                q.clone() * (src_val - query_val - diff_val.clone()),
                q.clone() * (diff_val.clone() * diff_val - sq_val.clone()),
                q * (acc_prev_val + sq_val - acc_val),
            ]
        });

        let q_prod = meta.selector();
        meta.create_gate("product update", |meta| {
            let q = meta.query_selector(q_prod);
            let idx = meta.query_advice(src, Rotation::cur());
            let dist = meta.query_advice(query, Rotation::cur());
            let val = meta.query_advice(diff, Rotation::cur());
            let factor = meta.query_advice(sq, Rotation::cur());
            let prev_prod = meta.query_advice(acc_prev, Rotation::cur());
            let prod = meta.query_advice(acc, Rotation::cur());
            let t = meta.query_advice(param_t, Rotation::cur());
            let r = meta.query_advice(param_r, Rotation::cur());
            vec![
                q.clone() * (idx * t.clone() + dist - val.clone()),
                q.clone() * (r - val - factor.clone()),
                q * (prev_prod * factor - prod),
            ]
        });

        let range = RangeCheckChip::configure(meta, param_t, param_r);

        BruteForceConfig {
            src,
            query,
            diff,
            sq,
            acc_prev,
            acc,
            param_t,
            param_r,
            q_calc,
            q_prod,
            range,
            instance,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        let range_chip = RangeCheckChip::construct(config.range.clone());

        // Assign query public inputs
        let query_cells: Vec<AssignedCell<Fr, Fr>> = layouter.assign_region(
            || "load query",
            |mut region: Region<'_, Fr>| {
                let mut offset = 0;
                let mut cells = Vec::with_capacity(self.query.len());
                for value in &self.query {
                    let cell = region.assign_advice(
                        || "query",
                        config.query,
                        offset,
                        || Value::known(*value),
                    )?;
                    cells.push(cell);
                    offset += 1;
                }
                Ok(cells)
            },
        )?;

        for (i, cell) in query_cells.iter().enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, i)?;
        }

        // Compute distances and collect final accumulators
        let mut dist_cells: Vec<AssignedCell<Fr, Fr>> = Vec::with_capacity(self.src_vecs.len());
        let mut dist_values: Vec<Fr> = Vec::with_capacity(self.src_vecs.len());
        layouter.assign_region(
            || "distance computation",
            |mut region: Region<'_, Fr>| {
                let mut offset = 0;
                for vec in &self.src_vecs {
                    let mut prev_acc_value = Fr::zero();
                    let mut prev_acc_cell: Option<AssignedCell<Fr, Fr>> = None;
                    for (j, src_val) in vec.iter().enumerate() {
                        let query_val = self.query[j];
                        let diff_val = *src_val - query_val;
                        let sq_val = diff_val.square();
                        let acc_val = prev_acc_value + sq_val;

                        region.assign_advice(
                            || "src",
                            config.src,
                            offset,
                            || Value::known(*src_val),
                        )?;
                        let query_cell = region.assign_advice(
                            || "query copy",
                            config.query,
                            offset,
                            || Value::known(query_val),
                        )?;
                        region.constrain_equal(query_cell.cell(), query_cells[j].cell())?;
                        region.assign_advice(
                            || "diff",
                            config.diff,
                            offset,
                            || Value::known(diff_val),
                        )?;
                        region.assign_advice(
                            || "sq",
                            config.sq,
                            offset,
                            || Value::known(sq_val),
                        )?;
                        let acc_prev_cell = region.assign_advice(
                            || "acc prev",
                            config.acc_prev,
                            offset,
                            || Value::known(prev_acc_value),
                        )?;
                        if let Some(prev) = &prev_acc_cell {
                            region.constrain_equal(acc_prev_cell.cell(), prev.cell())?;
                        }
                        let acc_cell = region.assign_advice(
                            || "acc",
                            config.acc,
                            offset,
                            || Value::known(acc_val),
                        )?;

                        config.q_calc.enable(&mut region, offset)?;

                        prev_acc_value = acc_val;
                        prev_acc_cell = Some(acc_cell.clone());

                        if j + 1 == vec.len() {
                            dist_cells.push(acc_cell.clone());
                            dist_values.push(acc_val);
                        }

                        offset += 1;
                    }
                }
                Ok(())
            },
        )?;

        // Range checks for sorted distances
        for i in 0..self.sorted_pairs_u64.len().saturating_sub(1) {
            let (_, curr_dis) = self.sorted_pairs_u64[i];
            let (_, next_dis) = self.sorted_pairs_u64[i + 1];
            let diff = next_dis.checked_sub(curr_dis).ok_or(Error::Synthesis)?;
            range_chip.assign(&mut layouter, diff)?;
        }

        // Product for unsorted set
        let unsorted_final = layouter.assign_region(
            || "unsorted product",
            |mut region: Region<'_, Fr>| {
                let mut offset = 0;
                let mut prev_prod_val = Fr::one();
                let mut prev_prod_cell: Option<AssignedCell<Fr, Fr>> = None;
                let mut prev_t_cell: Option<AssignedCell<Fr, Fr>> = None;
                let mut prev_r_cell: Option<AssignedCell<Fr, Fr>> = None;

                for (i, dist_cell) in dist_cells.iter().enumerate() {
                    let dist_val = dist_values[i];
                    let idx_val = Fr::from(i as u64);
                    let val_val = idx_val * self.fs_t + dist_val;
                    let factor_val = self.fs_r - val_val;
                    let prod_val = prev_prod_val * factor_val;

                    region.assign_advice(|| "idx", config.src, offset, || Value::known(idx_val))?;
                    let dist_copy = region.assign_advice(
                        || "dist copy",
                        config.query,
                        offset,
                        || Value::known(dist_val),
                    )?;
                    region.constrain_equal(dist_copy.cell(), dist_cell.cell())?;
                    region.assign_advice(
                        || "val",
                        config.diff,
                        offset,
                        || Value::known(val_val),
                    )?;
                    region.assign_advice(
                        || "factor",
                        config.sq,
                        offset,
                        || Value::known(factor_val),
                    )?;
                    let prev_cell = region.assign_advice(
                        || "prev prod",
                        config.acc_prev,
                        offset,
                        || Value::known(prev_prod_val),
                    )?;
                    if let Some(prev) = &prev_prod_cell {
                        region.constrain_equal(prev_cell.cell(), prev.cell())?;
                    }
                    let prod_cell = region.assign_advice(
                        || "prod",
                        config.acc,
                        offset,
                        || Value::known(prod_val),
                    )?;
                    let t_cell = region.assign_advice(
                        || "t",
                        config.param_t,
                        offset,
                        || Value::known(self.fs_t),
                    )?;
                    if let Some(prev_t) = &prev_t_cell {
                        region.constrain_equal(t_cell.cell(), prev_t.cell())?;
                    }
                    prev_t_cell = Some(t_cell);
                    let r_cell = region.assign_advice(
                        || "r",
                        config.param_r,
                        offset,
                        || Value::known(self.fs_r),
                    )?;
                    if let Some(prev_r) = &prev_r_cell {
                        region.constrain_equal(r_cell.cell(), prev_r.cell())?;
                    }
                    prev_r_cell = Some(r_cell);

                    config.q_prod.enable(&mut region, offset)?;

                    prev_prod_val = prod_val;
                    prev_prod_cell = Some(prod_cell.clone());
                    offset += 1;
                }

                prev_prod_cell.ok_or(Error::Synthesis)
            },
        )?;

        // Product for sorted set
        let sorted_final = layouter.assign_region(
            || "sorted product",
            |mut region: Region<'_, Fr>| {
                let mut offset = 0;
                let mut prev_prod_val = Fr::one();
                let mut prev_prod_cell: Option<AssignedCell<Fr, Fr>> = None;
                let mut prev_t_cell: Option<AssignedCell<Fr, Fr>> = None;
                let mut prev_r_cell: Option<AssignedCell<Fr, Fr>> = None;

                for (idx_u64, dis_u64) in &self.sorted_pairs_u64 {
                    let idx_val = Fr::from(*idx_u64);
                    let dist_val = Fr::from(*dis_u64);
                    let val_val = idx_val * self.fs_t + dist_val;
                    let factor_val = self.fs_r - val_val;
                    let prod_val = prev_prod_val * factor_val;

                    region.assign_advice(|| "idx", config.src, offset, || Value::known(idx_val))?;
                    region.assign_advice(
                        || "dist",
                        config.query,
                        offset,
                        || Value::known(dist_val),
                    )?;
                    region.assign_advice(
                        || "val",
                        config.diff,
                        offset,
                        || Value::known(val_val),
                    )?;
                    region.assign_advice(
                        || "factor",
                        config.sq,
                        offset,
                        || Value::known(factor_val),
                    )?;
                    let prev_cell = region.assign_advice(
                        || "prev prod",
                        config.acc_prev,
                        offset,
                        || Value::known(prev_prod_val),
                    )?;
                    if let Some(prev) = &prev_prod_cell {
                        region.constrain_equal(prev_cell.cell(), prev.cell())?;
                    }
                    let prod_cell = region.assign_advice(
                        || "prod",
                        config.acc,
                        offset,
                        || Value::known(prod_val),
                    )?;
                    let t_cell = region.assign_advice(
                        || "t",
                        config.param_t,
                        offset,
                        || Value::known(self.fs_t),
                    )?;
                    if let Some(prev_t) = &prev_t_cell {
                        region.constrain_equal(t_cell.cell(), prev_t.cell())?;
                    }
                    prev_t_cell = Some(t_cell);
                    let r_cell = region.assign_advice(
                        || "r",
                        config.param_r,
                        offset,
                        || Value::known(self.fs_r),
                    )?;
                    if let Some(prev_r) = &prev_r_cell {
                        region.constrain_equal(r_cell.cell(), prev_r.cell())?;
                    }
                    prev_r_cell = Some(r_cell);

                    config.q_prod.enable(&mut region, offset)?;

                    prev_prod_val = prod_val;
                    prev_prod_cell = Some(prod_cell.clone());
                    offset += 1;
                }

                prev_prod_cell.ok_or(Error::Synthesis)
            },
        )?;

        layouter.constrain_equal(unsorted_final.cell(), sorted_final.cell())?;

        Ok(())
    }
}
