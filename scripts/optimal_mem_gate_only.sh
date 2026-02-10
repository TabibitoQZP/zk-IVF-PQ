python -m bench.optimal_mem_gate_only \
  --merkled \
  --D 128 \
  --n-list 8192 \
  --n-probe 64 \
  --n 256

python -m bench.optimal_mem_gate_only \
  --merkled \
  --D 960 \
  --n-list 8192 \
  --n-probe 64 \
  --n 256

python -m bench.optimal_mem_gate_only \
  --merkled \
  --D 512 \
  --n-list 128 \
  --n-probe 1 \
  --n 8192

python -m bench.optimal_mem_gate_only \
  --merkled \
  --D 512 \
  --n-list 256 \
  --n-probe 2 \
  --n 4096

python -m bench.optimal_mem_gate_only \
  --merkled \
  --D 512 \
  --n-list 512 \
  --n-probe 4 \
  --n 2048

python -m bench.optimal_mem_gate_only \
  --merkled \
  --D 512 \
  --n-list 1024 \
  --n-probe 8 \
  --n 1024




python -m bench.optimal_mem_gate_only \
  --merkled \
  --D 384 \
  --n-list 512 \
  --n-probe 1 \
  --n 32768

python -m bench.optimal_mem_gate_only \
  --merkled \
  --D 384 \
  --n-list 1024 \
  --n-probe 2 \
  --n 16384

python -m bench.optimal_mem_gate_only \
  --merkled \
  --D 384 \
  --n-list 2048 \
  --n-probe 4 \
  --n 8192

python -m bench.optimal_mem_gate_only \
  --merkled \
  --D 384 \
  --n-list 4096 \
  --n-probe 8 \
  --n 4096