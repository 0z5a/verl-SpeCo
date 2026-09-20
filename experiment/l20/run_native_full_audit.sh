#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/0z5a/speco-l20-20260919
run=$root/c5-native-latest
scripts=$root/variants/pr10-native/experiment/l20
export SPECO_GPUS=${SPECO_GPUS:-2,3} SPECO_PYTHON_CACHE=/dev/shm/speco-native-python-20260920
export SPECO_AUDIT_OVERLAY=$run/pr10-retention-overlay-v2 SPECO_VLLM_OVERLAY=$run/public-loader-overlay
export SPECO_PADDING_OVERLAY=$run/padding-overlay
eagle_run=${EAGLE_RUN:-v16}
dflash_run=${DFLASH_RUN:-v9}
common=(actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=9216
        actor_rollout_ref.rollout.drafter.training.draft_update_weights_bucket_megabytes=128
        actor_rollout_ref.rollout.drafter.training.draft_update_use_shm=True)
status=0
timeout -k 30 3600 bash "$scripts/run_native_online.sh" \
  actor_rollout_ref.rollout.drafter.model_path="$run/eagle-training-alias" \
  +actor_rollout_ref.rollout.drafter.vllm.speculative_config_overrides.model="$root/models/eagle" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=134217728 \
  actor_rollout_ref.rollout.max_num_batched_tokens=256 "${common[@]}" \
  > "$run/evidence/full-e2e/online-eagle-pr10-$eagle_run.log" 2>&1 || status=$?
echo "$status" > "$run/evidence/full-e2e/online-eagle-pr10-$eagle_run.exit"
status=0
timeout -k 30 3600 bash "$scripts/run_native_dflash.sh" "${common[@]}" \
  > "$run/evidence/full-e2e/online-dflash-pr10-$dflash_run.log" 2>&1 || status=$?
echo "$status" > "$run/evidence/full-e2e/online-dflash-pr10-$dflash_run.exit"
