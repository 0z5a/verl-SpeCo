#!/usr/bin/env bash
set -euo pipefail
root=${1:?native C5 working directory}
py=${2:?native environment Python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ALLOW_INSECURE_SERIALIZATION=1
cd "$root"
# Native vLLM 0.18 uses pard_token for the parallel-drafting mask token ID.
"$py" - <<'PY'
import json
from pathlib import Path
p = Path('models/draft-eagle3/config.json')
config = json.loads(p.read_text())
config['pard_token'] = config['mask_token_id']
# v0.18 reads auxiliary collection layers from the top-level config.
config['eagle_aux_hidden_state_layer_ids'] = [1, 2, 3]
p.write_text(json.dumps(config, indent=2) + '\n')
PY
for label in native-unpatched native-candidate; do
  flags=(--legacy-worker-accessor)
  if [ "$label" = native-unpatched ]; then
    flags+=(--expect-failure)
    export PYTHONPATH=
  else
    export PYTHONPATH="$root/overlay"
  fi
  if timeout -k 15 300 "$py" check_public_loader_update.py --models-root "$root/models" --evidence-root "$root/evidence" --label "$label" "${flags[@]}" > "evidence/$label.log" 2>&1; then
    echo 0 > "evidence/$label.exit"
  else
    status=$?
    echo "$status" > "evidence/$label.exit"
    exit "$status"
  fi
done
