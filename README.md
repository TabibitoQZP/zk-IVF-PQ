# zk-IVF-PQ

## Technical Note

Implementation details are provided in our [technical note](Technical_Details_of_V3DB.pdf).

## Build

Build and install the Python extension:

```bash
maturin develop --release
```

## Experiment 1: Retrieval Utility Evaluation

### Classic ANN (SIFT1M / GIST1M)

```bash
bash scripts/acc_bench.sh
```

Results are cached under `data/acc_bench/`.

### IR (MS MARCO passage retrieval, dev)

Run the two evaluation jobs (high-acc and zk-opt) and then aggregate metrics:

```bash
python -m bench.ms_macro_eval \
  --num-runs 1 \
  --n-list 8192 \
  --n-probe 64 \
  --M 8 \
  --K 256 \
  --cluster-bound 2048 \
  --top-k 1000 \
  --out-dir data/exp1/ms_macro_high_acc

python -m bench.ms_macro_eval \
  --num-runs 1 \
  --n-list 2048 \
  --n-probe 16 \
  --M 8 \
  --K 256 \
  --cluster-bound 8192 \
  --top-k 1000 \
  --out-dir data/exp1/ms_macro_zk_fast

python -m bench.ms_macro_result data/exp1/ms_macro_high_acc data/exp1/ms_macro_zk_fast
```

Outputs are written under `data/exp1/`.

## Experiment 2: Proof Cost Evaluation

Script: `scripts/bench_suite.sh` (runs `python -m bench.bench_suite` and caches results under `data/bench_result/`).

```bash
bash scripts/bench_suite.sh
```

Optional: if `matplotlib` is available, a summary plot is generated at `data/bench_result/bench_summary.pdf`.

## Experiment 3: Configuration Trade-offs

### Fixed scan budget: sweep `n_list`

```bash
bash scripts/optimal_config.sh
```

Outputs are written to `data/optimal_config/`.

### Fixed code budget: gate-only sweep over `K` (and zk-opt selection)

```bash
bash scripts/optimal_mem_gate_only_ann_ir.sh
```

Outputs are written to `data/optimal_mem_gate_only/`.
