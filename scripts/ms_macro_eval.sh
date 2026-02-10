python -m bench.ms_macro_eval \
  --num-runs 1 \
  --n-list 4096 \
  --n-probe 64 \
  --M 8 \
  --K 256 \
  --cluster-bound 4096 \
  --top-k 1000 \
  --out-dir data/ms_macro_eval_high_acc

python -m bench.ms_macro_eval \
  --num-runs 1 \
  --n-list 1024 \
  --n-probe 2 \
  --M 8 \
  --K 256 \
  --cluster-bound 16384 \
  --top-k 1000 \
  --out-dir data/ms_macro_eval_zk_fast

python -m bench.ms_macro_eval \
  --num-runs 1 \
  --n-list 8192 \
  --n-probe 256 \
  --M 8 \
  --K 256 \
  --cluster-bound 2048 \
  --top-k 1000 \
  --out-dir data/ms_macro_eval_extra_high_acc
