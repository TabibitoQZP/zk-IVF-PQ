#!/usr/bin/env bash
set -euo pipefail

# Experiment 1 / Classic ANN (Table "Experiment 1 datasets and IVF-PQ layouts")
# SIFT1M high-acc: (n_list, n_probe, n) = (8192, 64, 256)
CUDA_VISIBLE_DEVICES=7 python -m tests.acc_bench \
  --data-name sift \
  --name gpu4_exp1_sift_high_acc_multik \
  --num-runs 1 \
  --M 8 \
  --K 256 \
  --top-k 100 \
  --report-ks 1,10,50,100,500 \
  --scale-n 65536 \
  --n-list 8192 \
  --n-probe 64 \
  --cluster-bound 256 \
  --layout none

sleep 60

# GIST1M high-acc: (n_list, n_probe, n) = (8192, 64, 256)
CUDA_VISIBLE_DEVICES=7 python -m tests.acc_bench \
  --data-name gist \
  --name gpu4_exp1_gist_high_acc_multik \
  --num-runs 1 \
  --M 8 \
  --K 256 \
  --top-k 100 \
  --report-ks 1,10,50,100,500 \
  --scale-n 65536 \
  --n-list 8192 \
  --n-probe 64 \
  --cluster-bound 256 \
  --layout none

sleep 60

# SIFT1M zk-opt: (n_list, n_probe, n) = (1024, 8, 2048)
CUDA_VISIBLE_DEVICES=7 python -m tests.acc_bench \
  --data-name sift \
  --name gpu4_exp1_sift_zk_opt_multik \
  --num-runs 1 \
  --M 8 \
  --K 256 \
  --top-k 100 \
  --report-ks 1,10,50,100,500 \
  --scale-n 65536 \
  --n-list 1024 \
  --n-probe 8 \
  --cluster-bound 2048 \
  --layout none

sleep 60

# GIST1M zk-opt: (n_list, n_probe, n) = (512, 4, 4096)
CUDA_VISIBLE_DEVICES=7 python -m tests.acc_bench \
  --data-name gist \
  --name gpu4_exp1_gist_zk_opt_multik \
  --num-runs 1 \
  --M 8 \
  --K 256 \
  --top-k 100 \
  --report-ks 1,10,50,100,500 \
  --scale-n 65536 \
  --n-list 512 \
  --n-probe 4 \
  --cluster-bound 4096 \
  --layout none
