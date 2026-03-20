#!/usr/bin/env bash
# 如需排查问题可以打开下一行
set -x

# 配置主节点 IP
MASTER_IP="填写当前节点IP"
# 配置当前节点用于通信的网卡名。在宿主机上执行`ifconfig`，查询本机IP所对应的网络接口名
SOCKET_IFNAME="填写当前节点通信网卡名"
CURRENT_IP="填写当前节点IP"
echo "MASTER_IP = $MASTER_IP"
echo "CURRENT_IP = $CURRENT_IP"
echo "SOCKET_IFNAME = $SOCKET_IFNAME"

ulimit -n 32768
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_SOCKET_IFNAME="$SOCKET_IFNAME"
export GLOO_SOCKET_IFNAME="$SOCKET_IFNAME"
export HCCL_EXEC_TIMEOUT=1800
export HCCL_CONNECT_TIMEOUT=1800
export VLLM_USE_V1=1
export VLLM_VERSION=0.11.0
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_IF_BASE_PORT=48890
export RAY_DEBUG_POST_MORTEM=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:2048
# 配置cann路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 配置集群规模
NNODES=1          # 总节点数（主 + 从）
NPUS_PER_NODE=8   # 每台机器的 NPU 数

# 配置huggingface格式权重路径，转化为megatron格式的权重路径，训练数据集路径，测试数据集路径，保存权重的路径
HF_MODEL_PATH=/home/ori_models/Qwen3-8B
DIST_CKPT_PATH=/home/bridge/Qwen3-8B/iter_0000000
TRAIN_DATA_PATH=/home/datasets/processed_gsm8k/train.parquet
TEST_DATA_PATH=/home/datasets/processed_gsm8k/test.parquet
SAVE_CKPT_PATH=/home/checkpoints/verl_grpo_example_gsm8k/qwen3_8b_function_bridge

tp_rollout=4
tp=2
pp=2
cp=2

run_training() {
    python3 -m verl.trainer.main_ppo --config-path=config \
        --config-name='ppo_megatron_trainer.yaml' \
        algorithm.adv_estimator=grpo \
        data.train_files=$TRAIN_DATA_PATH \
        data.val_files=$TEST_DATA_PATH \
        data.train_batch_size=8 \
        data.max_prompt_length=2048 \
        data.max_response_length=4096 \
        data.filter_overlong_prompts=True \
        data.truncation='left' \
        actor_rollout_ref.model.path=$HF_MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=1e-7 \
        actor_rollout_ref.actor.ppo_mini_batch_size=1 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$tp_rollout \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.project_name='verl_grpo_example_gsm8k' \
        trainer.experiment_name='qwen3_8b_function_bridge' \
        trainer.n_gpus_per_node=$NPUS_PER_NODE \
        trainer.nnodes=$NNODES \
        trainer.save_freq=100 \
        trainer.default_local_dir=$SAVE_CKPT_PATH \
        trainer.test_freq=10 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=400 \
        trainer.resume_mode=disable \
        trainer.device=npu \
        actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
        actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
        actor_rollout_ref.actor.checkpoint.save_contents=['model'] \
        actor_rollout_ref.actor.megatron.use_mbridge=True \
        actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$tp \
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$tp \
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$pp \
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$pp \
        actor_rollout_ref.actor.megatron.context_parallel_size=$cp \
        +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=$cp \
        +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo=kvallgather_cp_algo \
        +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block \
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=8 \
        +actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True \
        +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True
}

pkill -9 python
ray stop --force
rm -rf /tmp/ray

# 配置ray端口
RAY_PORT=6344
ray start \
  --head \
  --port $RAY_PORT \
  --dashboard-host="$MASTER_IP" \
  --node-ip-address="$CURRENT_IP" \
  --dashboard-port=8260 \
  --resources="{\"NPU\": $NPUS_PER_NODE}"

run_training
