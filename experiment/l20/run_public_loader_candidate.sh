#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/experiment/online-deps:/experiment/variants/peagle-serving
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1
py=/experiment/.venv-clean/bin/python
out=/experiment/evidence/l20-20260919
source_file=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/llama_eagle3.py
backup=$(mktemp)
cp "$source_file" "$backup"
trap 'cp "$backup" "$source_file"; rm -f "$backup"' EXIT
cd /usr/local/lib/python3.12/dist-packages
# Verify the exact one-line candidate against the installed version before testing.
patch --dry-run -p1 < /experiment/vllm-eagle3-idempotent-prefix.patch
timeout 600 "$py" /experiment/check_public_loader_update.py --expect-failure --label public-loader-unpatched > "$out/public-loader-unpatched.log" 2>&1
patch -p1 < /experiment/vllm-eagle3-idempotent-prefix.patch
timeout 600 "$py" /experiment/check_public_loader_update.py --label public-loader-local-candidate > "$out/public-loader-local-candidate.log" 2>&1
