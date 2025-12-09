#!/usr/bin/env bash
set -euo pipefail
# 如需排查问题可以打开下一行
#set -x

# 配置主节点 IP
MASTER_IP="填写主节点IP"
# 配置当前节点用于通信的网卡名。在宿主机上执行`ifconfig`，查询本机IP所对应的网络接口名
SOCKET_IFNAME="填写当前节点通信网卡名"
CURRENT_IP=$(ifconfig "$SOCKET_IFNAME" | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
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
NNODES=2          # 总节点数（主 + 从）
NPUS_PER_NODE=16  # 每台机器的 NPU 数

# 配置huggingface格式权重路径，转化为megatron格式的权重路径，训练数据集路径，测试数据集路径，保存权重的路径
HF_MODEL_PATH=/weight/Qwen3-30B-A3B
DIST_CKPT_PATH=/weight/Qwen3-30B-A3B-dist
TRAIN_DATA_PATH=/data/processed_gsm8k/train.parquet
TEST_DATA_PATH=/data/processed_gsm8k/test.parquet
SAVE_CKPT_PATH=/data/checkpoints/verl_grpo_example_gsm8k_math_base/qwen3_30b_moe_megatron_base

tp=4
pp=2
cp=2
ep=8
etp=1

run_training() {
    python3 -m verl.trainer.main_ppo --config-path=config \
        --config-name='ppo_megatron_trainer.yaml'\
        algorithm.adv_estimator=grpo \
        data.train_files=$TRAIN_DATA_PATH \
        data.val_files=$TEST_DATA_PATH \
        data.train_batch_size=16 \
        data.max_prompt_length=1024 \
        data.max_response_length=2048 \
        data.filter_overlong_prompts=True \
        data.truncation='left' \
        actor_rollout_ref.model.path=$HF_MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=16 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$pp \
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$tp \
        actor_rollout_ref.actor.megatron.expert_model_parallel_size=$ep \
        actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
        actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$tp \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$pp \
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$tp \
        actor_rollout_ref.ref.megatron.expert_model_parallel_size=$ep \
        actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
        actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.project_name='verl_grpo_example_gsm8k_math_base' \
        trainer.experiment_name='qwen3_30b_moe_megatron_base' \
        trainer.n_gpus_per_node=$NPUS_PER_NODE \
        trainer.nnodes=$NNODES \
        trainer.save_freq=20 \
        trainer.default_local_dir=$SAVE_CKPT_PATH \
        trainer.test_freq=20 \
        trainer.total_epochs=10 \
        actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$etp \
        actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$etp \
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
        actor_rollout_ref.actor.megatron.context_parallel_size=$cp \
        actor_rollout_ref.ref.megatron.param_offload=True \
        actor_rollout_ref.actor.megatron.param_offload=True \
        actor_rollout_ref.actor.megatron.optimizer_offload=True \
        actor_rollout_ref.actor.megatron.grad_offload=True \
        trainer.device=npu \
        +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
        +actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True \
        +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True \
        +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=$cp
}

pkill -9 python
ray stop --force
rm -rf /tmp/ray

# 配置ray端口
RAY_PORT=6344
if [ "$MASTER_IP" = "$CURRENT_IP" ]; then
  ray start \
    --head \
    --port $RAY_PORT \
    --dashboard-host="$MASTER_IP" \
    --node-ip-address="$CURRENT_IP" \
    --dashboard-port=8260 \
    --resources="{\"NPU\": $NPUS_PER_NODE}"

  # 等待所有节点成功注册到 Ray 集群
  while true; do
      ray_status_output=$(ray status)

      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / NPUS_PER_NODE))

      if [ "$device_count" -eq "$NNODES" ]; then
          echo "[INFO] Ray cluster is ready with $device_count nodes (from $npu_count NPU resources)."
          ray status

          # 在主节点上启动强化学习训练
          run_training
          break
      else
          echo "[INFO] Waiting for Ray to allocate $NNODES nodes. Current node count: $device_count"
          sleep 5
      fi
  done
else
  while true; do
      # 子节点不断尝试往主节点注册
      ray start \
        --address="$MASTER_IP:$RAY_PORT" \
        --resources="{\"NPU\": $NPUS_PER_NODE}" \
        --node-ip-address="$CURRENT_IP"

      # 简单用 ray status 判断是否连上
      if ray status > /dev/null 2>&1; then
          echo "[INFO] Successfully connected to the Ray cluster from $CURRENT_IP!"
          break
      else
          echo "[WARN] Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi

sleep 600
