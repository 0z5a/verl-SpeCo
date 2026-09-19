#!/usr/bin/env bash
set -euo pipefail
cd /experiment
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
export HF_HOME=/experiment/.cache/huggingface
RUN_DIR=${RUN_DIR:-evidence/l20-20260919/final}
mkdir -p "$RUN_DIR"
for tokens in 256 1024; do
  for arm in A0 P0 P1 A1; do
    source=/experiment
    partitions=1
    if [[ "$arm" == P* ]]; then
      source=/experiment/variants/patch
      partitions=2
    fi
    .venv-clean/bin/python benchmark_training.py --source "$source" \
      --target /experiment/models/target --tokens "$tokens" --partitions "$partitions" \
      --output "$RUN_DIR/train-${tokens}-${arm}.json" \
      > "$RUN_DIR/train-${tokens}-${arm}.log" 2>&1
  done
done
