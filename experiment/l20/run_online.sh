#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd /experiment
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export HF_HOME=/experiment/.cache/huggingface
export RAY_TMPDIR=/experiment/.cache/ray
export TORCHINDUCTOR_CACHE_DIR=/experiment/.cache/inductor
export TRITON_CACHE_DIR=/experiment/.cache/triton
export HYDRA_FULL_ERROR=1
export PYTHONSAFEPATH=1
export PYTHONPATH=/experiment/online-deps:${SPECO_SOURCE:-/experiment}:/experiment
/experiment/.venv-clean/bin/python "$SCRIPT_DIR/prepare_data.py"
cd "${SPECO_SOURCE:-/experiment}"
/experiment/.venv-clean/bin/python -m verl_speco.main \
 algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
 ray_kwargs.ray_init.num_cpus=16 \
 data.train_files=/experiment/evidence/l20-20260919/train.parquet \
 data.val_files=/experiment/evidence/l20-20260919/val.parquet \
 data.train_batch_size=2 data.max_prompt_length=128 data.max_response_length=64 \
 data.filter_overlong_prompts_workers=1 data.truncation=error \
 actor_rollout_ref.model.path=/experiment/models/target \
 actor_rollout_ref.model.use_remove_padding=True \
 +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.strategy=fsdp2 \
 actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=2 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.actor.use_kl_loss=False \
 actor_rollout_ref.actor.calculate_entropy=False \
 actor_rollout_ref.actor.fsdp_config.param_offload=True \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.load_format=auto \
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
 actor_rollout_ref.rollout.n=2 \
 actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=10240 \
 actor_rollout_ref.rollout.max_model_len=256 \
 actor_rollout_ref.rollout.max_num_seqs=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.enforce_eager=True \
 actor_rollout_ref.rollout.agent.num_workers=2 \
 actor_rollout_ref.rollout.drafter.enable=True \
 actor_rollout_ref.rollout.drafter.enable_drafter_training=True \
 actor_rollout_ref.rollout.drafter.model_path=/experiment/models/eagle \
 actor_rollout_ref.rollout.drafter.speculative_algorithm=EAGLE3 \
 actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_sgl=False \
 actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=True \
 actor_rollout_ref.rollout.drafter.training.old_logprob_hidden_capture_impl=forward_hook \
 actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=32 \
 actor_rollout_ref.rollout.drafter.training.hidden_state_window_min_rows=32 \
 actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.drafter.training.step=1 \
 actor_rollout_ref.rollout.drafter.training.collect_interval_steps=1 \
 actor_rollout_ref.rollout.drafter.training.training_interval_steps=1 \
 actor_rollout_ref.rollout.drafter.training.publish_interval_steps=1 \
 actor_rollout_ref.rollout.drafter.training.publish_async=True \
 speco.runtime.weight_update_mode=compat \
 trainer.n_gpus_per_node=2 trainer.nnodes=1 \
 trainer.val_before_train=False trainer.logger='[console]' \
 trainer.save_freq=-1 trainer.test_freq=-1 trainer.total_training_steps=20 \
 trainer.total_epochs=3 trainer.project_name=speco-l20 \
 trainer.experiment_name=eagle3-compat \
 trainer.default_local_dir=/experiment/evidence/l20-20260919/checkpoints "$@"
