#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/zipengqiu/miniconda3/envs/zk/bin/python"
GATE_SCRIPT="$SCRIPT_DIR/../bench/gate_count.py"

echo "[1/2] MSMARCO-like config: n_list=8192, n_probe=64, n=2048"
"$PYTHON_BIN" "$GATE_SCRIPT" \
  --merkled \
  --D 384 \
  --n-list 8192 \
  --n-probe 64 \
  --n 2048 \
  --top-k 64 \
  --M 8 \
  --K 256

echo
echo "[2/2] MSMARCO-like config: n_list=2048, n_probe=16, n=8192"
"$PYTHON_BIN" "$GATE_SCRIPT" \
  --merkled \
  --D 384 \
  --n-list 2048 \
  --n-probe 16 \
  --n 8192 \
  --top-k 64 \
  --M 8 \
  --K 256
