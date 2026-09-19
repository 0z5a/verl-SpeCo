#!/usr/bin/env bash
set -eu
cd /experiment
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
export HF_HOME=/experiment/.cache/huggingface
export XDG_CACHE_HOME=/experiment/.cache
export TORCHINDUCTOR_CACHE_DIR=/experiment/.cache/inductor
export TRITON_CACHE_DIR=/experiment/.cache/triton
export PYTHONPATH=/experiment/online-deps:/experiment
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

if [ "$#" -eq 0 ]; then
    set -- baseline draft-original draft-mapped draft-eagle3
fi
for mode in "$@"; do
    if .venv-clean/bin/python check_tiny_peagle_serving.py "$mode" \
        > "evidence/l20-20260919/tiny-peagle-$mode.log" 2>&1; then
        status=0
    else
        status=$?
    fi
    printf '%s\n' "$status" > "evidence/l20-20260919/tiny-peagle-$mode.exit"
done
