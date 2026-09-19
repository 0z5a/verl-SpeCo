#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
bash "$SCRIPT_DIR/run_online.sh" \
 actor_rollout_ref.rollout.drafter.model_path=/experiment/models/dflash \
 actor_rollout_ref.rollout.drafter.speculative_algorithm=DFLASH \
 actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=16 \
 actor_rollout_ref.rollout.drafter.training.dflash_num_anchors=8 \
 actor_rollout_ref.rollout.drafter.training.dflash_max_window=32 \
 trainer.experiment_name=dflash-compat "$@"
