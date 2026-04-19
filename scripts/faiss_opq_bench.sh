#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/zipengqiu/miniconda3/envs/zk/bin/python"

# FAISS OPQ + IVF-PQ baseline, matched to the main Experiment 1 layouts.
# The benchmark itself will fall back to CPU if faiss-gpu is unavailable.

# SIFT1M high-acc: (n_list, n_probe) = (8192, 64)
CUDA_VISIBLE_DEVICES=6 "$PYTHON_BIN" -m tests.faiss_opq_bench \
  --data-name sift \
  --name faiss_opq_sift_high_acc_mod8 \
  --num-runs 1 \
  --M 8 \
  --K 256 \
  --top-k 100 \
  --report-ks 1,10,50,100 \
  --n-list 8192 \
  --n-probe 64 \
  --layout mod8 \
  --use-gpu \
  --gpu-device 0

sleep 60

# GIST1M high-acc: (n_list, n_probe) = (8192, 64)
CUDA_VISIBLE_DEVICES=6 "$PYTHON_BIN" -m tests.faiss_opq_bench \
  --data-name gist \
  --name faiss_opq_gist_high_acc_mod8 \
  --num-runs 1 \
  --M 8 \
  --K 256 \
  --top-k 100 \
  --report-ks 1,10,50,100 \
  --n-list 8192 \
  --n-probe 64 \
  --layout mod8 \
  --use-gpu \
  --gpu-device 0

sleep 60

# SIFT1M zk-opt layout config: (n_list, n_probe) = (1024, 8)
CUDA_VISIBLE_DEVICES=6 "$PYTHON_BIN" -m tests.faiss_opq_bench \
  --data-name sift \
  --name faiss_opq_sift_zk_opt_mod8 \
  --num-runs 1 \
  --M 8 \
  --K 256 \
  --top-k 100 \
  --report-ks 1,10,50,100 \
  --n-list 1024 \
  --n-probe 8 \
  --layout mod8 \
  --use-gpu \
  --gpu-device 0

sleep 60

# GIST1M zk-opt layout config: (n_list, n_probe) = (512, 4)
CUDA_VISIBLE_DEVICES=6 "$PYTHON_BIN" -m tests.faiss_opq_bench \
  --data-name gist \
  --name faiss_opq_gist_zk_opt_mod8 \
  --num-runs 1 \
  --M 8 \
  --K 256 \
  --top-k 100 \
  --report-ks 1,10,50,100 \
  --n-list 512 \
  --n-probe 4 \
  --layout mod8 \
  --use-gpu \
  --gpu-device 0
