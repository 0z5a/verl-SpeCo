#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
root=/home/kxqandccx/0z5a/speco-l20-20260919
bash "$SCRIPT_DIR/run_native_online.sh" \
 actor_rollout_ref.rollout.drafter.model_path="$root/models/dflash" \
 actor_rollout_ref.rollout.drafter.speculative_algorithm=DFLASH \
 actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=16 \
 actor_rollout_ref.rollout.drafter.training.dflash_num_anchors=8 \
 actor_rollout_ref.rollout.drafter.training.dflash_max_window=32 \
 +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=134217728 \
 actor_rollout_ref.rollout.max_num_batched_tokens=256 \
 trainer.experiment_name=dflash-pr10-native \
 trainer.default_local_dir="$root/c5-native-latest/evidence/full-e2e/checkpoints-dflash-pr10" "$@"
