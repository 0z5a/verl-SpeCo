#!/usr/bin/env bash
set -euo pipefail
root=/experiment/c5-native-latest
fixture=$root/full-e2e-head64
out=$root/evidence/full-e2e
export PYTHONDONTWRITEBYTECODE=1
TIMEFORMAT='%R'
for arm in flat partition veomni; do
  engine=fsdp
  source=/experiment/variants/partition-before
  partition=1
  if [[ "$arm" == partition ]]; then partition=2; source=/experiment/variants/partition-e2e; fi
  if [[ "$arm" == veomni ]]; then engine=veomni; source=/experiment/variants/veomni; fi
  extra=()
  if [[ "$arm" != veomni ]]; then extra=(actor_rollout_ref.rollout.drafter.training.peagle_sequence_partitions="$partition"); fi
  { time SPECO_SOURCE="$source" timeout -k 15 600 bash /experiment/run_partition_standalone.sh "$engine" "native-head64-$arm" \
    actor_rollout_ref.model.path="$fixture/target" \
    actor_rollout_ref.rollout.drafter.model_path="$fixture/draft-original" \
    actor_rollout_ref.rollout.drafter.training.feature_store.path="$fixture/features-packed" \
    "${extra[@]}" \
    >"$out/train-head64-v2-$arm.log" 2>&1; } 2>"$out/train-head64-v2-$arm.seconds"
done
