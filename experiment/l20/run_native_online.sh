#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd /home/kxqandccx/0z5a/speco-l20-20260919
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=${SPECO_GPUS:-6,7} VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0
export RAY_DEDUP_LOGS=0
export RAY_worker_niceness=0
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export HF_HOME=/home/kxqandccx/0z5a/speco-l20-20260919/c5-native-latest/.cache/huggingface
export RAY_TMPDIR=/tmp/speco-pr10-ray
export TORCHINDUCTOR_CACHE_DIR=/home/kxqandccx/0z5a/speco-l20-20260919/c5-native-latest/.cache/inductor
export TRITON_CACHE_DIR=/home/kxqandccx/0z5a/speco-l20-20260919/c5-native-latest/.cache/triton
export HYDRA_FULL_ERROR=1
export PYTHONSAFEPATH=1
export PYTHONPATH=${SPECO_PYTHON_CACHE:+$SPECO_PYTHON_CACHE:}${SPECO_AUDIT_OVERLAY:+$SPECO_AUDIT_OVERLAY:}${SPECO_VLLM_OVERLAY:+$SPECO_VLLM_OVERLAY:}${SPECO_PADDING_OVERLAY:+$SPECO_PADDING_OVERLAY:}/home/kxqandccx/0z5a/speco-l20-20260919/c5-native-latest/deps-pr10:/home/kxqandccx/0z5a/speco-l20-20260919/variants/pr10-native
/home/kxqandccx/0z5a/speco-l20-20260919/c5-native-latest/.venv/bin/python "$SCRIPT_DIR/prepare_data.py"
cd "${SPECO_AUDIT_OVERLAY:-/home/kxqandccx/0z5a/speco-l20-20260919/variants/pr10-native}"
/home/kxqandccx/0z5a/speco-l20-20260919/c5-native-latest/.venv/bin/python -m verl_speco.main \
 algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
 ray_kwargs.ray_init.num_cpus=16 \
 data.train_files=/home/kxqandccx/0z5a/speco-l20-20260919/c5-native-latest/evidence/full-e2e/train.parquet \
 data.val_files=/home/kxqandccx/0z5a/speco-l20-20260919/c5-native-latest/evidence/full-e2e/val.parquet \
 data.train_batch_size=2 data.max_prompt_length=128 data.max_response_length=64 \
 data.filter_overlong_prompts_workers=1 data.truncation=error \
 actor_rollout_ref.model.path=/home/kxqandccx/0z5a/speco-l20-20260919/models/target \
 actor_rollout_ref.model.use_remove_padding=True \
 +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.strategy=fsdp2 \
 actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=2 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.actor.use_dynamic_bsz=False \
 actor_rollout_ref.actor.use_kl_loss=False \
 actor_rollout_ref.actor.calculate_entropy=False \
 actor_rollout_ref.actor.fsdp_config.param_offload=True \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.load_format=auto \
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.10 \
 actor_rollout_ref.rollout.n=2 \
 actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=128 \
 actor_rollout_ref.rollout.max_model_len=256 \
 actor_rollout_ref.rollout.max_num_seqs=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
 actor_rollout_ref.rollout.enforce_eager=True \
 actor_rollout_ref.rollout.agent.num_workers=2 \
 actor_rollout_ref.rollout.drafter.enable=True \
 actor_rollout_ref.rollout.drafter.enable_drafter_training=True \
 actor_rollout_ref.rollout.drafter.model_path=/home/kxqandccx/0z5a/speco-l20-20260919/models/eagle \
 actor_rollout_ref.rollout.drafter.speculative_algorithm=EAGLE3 \
 actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_sgl=False \
 actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=True \
 actor_rollout_ref.rollout.drafter.training.old_logprob_hidden_capture_impl=forward_hook \
 actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=32 \
 actor_rollout_ref.rollout.drafter.training.hidden_state_window_min_rows=2 \
 actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.drafter.training.step=1 \
 actor_rollout_ref.rollout.drafter.training.collect_interval_steps=1 \
 actor_rollout_ref.rollout.drafter.training.training_interval_steps=1 \
 actor_rollout_ref.rollout.drafter.training.publish_interval_steps=1 \
 actor_rollout_ref.rollout.drafter.training.publish_async=True \
 trainer.n_gpus_per_node=2 trainer.nnodes=1 \
 trainer.val_before_train=False trainer.logger='[console]' \
 custom_reward_function.path=/home/kxqandccx/0z5a/speco-l20-20260919/variants/pr10-native/experiment/l20/integration_reward.py \
 custom_reward_function.name=compute_score \
 +data.apply_chat_template_kwargs.enable_thinking=False \
 trainer.save_freq=10 trainer.test_freq=-1 trainer.total_training_steps=20 \
 trainer.total_epochs=3 trainer.project_name=speco-l20 \
 trainer.experiment_name=eagle3-compat \
 trainer.default_local_dir=/home/kxqandccx/0z5a/speco-l20-20260919/c5-native-latest/evidence/full-e2e/checkpoints-eagle-pr10 "$@"
