maturin develop --release

python -m bench.optimal_config \
  --N 65536 \
  --D 128 \
  --M 8 \
  --K 16 \
  --selected_count 8192 \
  --c 10 \
  --num-runs 5

python -m bench.optimal_config \
  --N 1048576 \
  --D 128 \
  --M 8 \
  --K 16 \
  --selected_count 8192 \
  --c 10 \
  --num-runs 5

python -m bench.optimal_config \
  --N 1048576 \
  --D 1024 \
  --M 32 \
  --K 256 \
  --selected_count 8192 \
  --c 6 \
  --num-runs 5
