python -m tests.acc_bench \
  --data-name sift \
  --num-runs 1 \
  --n-list 8192 \
  --n-probe 64 \
  --cluster-bound 256

python -m tests.acc_bench \
  --data-name gist \
  --num-runs 1 \
  --n-list 8192 \
  --n-probe 64 \
  --cluster-bound 256

python -m tests.acc_bench \
  --data-name sift \
  --num-runs 1 \
  --n-list 1024 \
  --n-probe 8 \
  --cluster-bound 2048

python -m tests.acc_bench \
  --data-name gist \
  --num-runs 1 \
  --n-list 512 \
  --n-probe 4 \
  --cluster-bound 4096

