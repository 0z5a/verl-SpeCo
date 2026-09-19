#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd /experiment
export UV_CACHE_DIR=/experiment/.cache/uv CUDA_HOME=/usr/local/cuda
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=80
export MAX_JOBS=${MAX_JOBS:-4}
uv pip install --python .venv-clean/bin/python --no-deps --no-build-isolation flash-attn==2.8.3
.venv-clean/bin/python "$SCRIPT_DIR/check_flash_attention.py"
