# Qwen3-30B-A3B 

## 概述

本文给出一个使用verl完成Qwen3-30B-A3B训练的示例

## 硬件环境

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 32 x Ascend NPUs |

注：本示例使用Atlas 800T A3双机进行训练，实测当前实验配置下，A2双机可能会出现OOM现象

## 软件环境

| 组件 | 版本 |
|------|------|
| Python | 3.11 |
| CANN | 8.5.0 |
| torch | 2.7.1 |
| torch_npu | 2.7.1 |
| vLLM | 0.11.0 |
| vLLM-ascend | 0.11.0rc1 |
| Megatron-LM | v0.12.1 |
| triton-ascend | 3.2.0rc4 |

## 模型训练

1. 准备数据

自行下载权重 [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/tree/main)

将huggingface格式的权重转换为Megatron Core格式：

```
# 权重转换脚本位于verl代码仓中
cd verl 

python scripts/converter_hf_to_mcore.py \ 
    --hf_model_path /weight/Qwen3-30B-A3B \ 
    --output_path /weight/Qwen3-30B-A3B-dist \ 
    --use_cpu_initialization
```

自行下载数据集 [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k/tree/main)


2. 配置训练脚本

* 首先根据需要调整如下路径：

`HF_MODEL_PATH` : Qwen3-30B-A3B 路径

`DIST_CKPT_PATH` : 转换后的权重路径

`data.train_files` `data.val_files` : 数据集路径

`trainer.default_local_dir` : 保存的checkpoints路径，当`trainer.save_freq=-1`（即不保存checkpoints）时可以不设置。需确保该路径下有足够的空间储存checkpoints。

* 其次根据需要调整其他参数，以下是几个示例：

`data.max_prompt_length` `data.max_response_length` : 分别用于设置输入提示的最大token数量，设置RL算法生成响应的最大token长度。增大时可以处理更长的提示，但会增加内存消耗和计算时间；减小时可以节省内存，但会使序列被截断，限制模型生成完整答案的能力。本示例中取值为 1024/2048，是在当前硬件资源约束下，在序列长度覆盖能力与显存/计算开销之间取得较好平衡的一组经验取值。

`actor_rollout_ref.actor.ppo_mini_batch_size` `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` : 分别决定了每次PPO更新时使用的样本总数，和单次前向传播的批次大小。约束：前者能被后者整除，且后者小于等于前者。在特定的硬件资源约束下，尽可能调大后者以提升训练效率。

* 最后，可以根据需要添加一些MindSpeed支持的高级特性，例如：

使用mbridge在线转换huggingface权重（需安装mbridge包）

```
actor_rollout_ref.actor.megatron.use_mbridge=True \
actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
actor_rollout_ref.ref.megatron.use_dist_checkpointing=False \
```

使用RoPE融合优化 [link](../../../docs/features/rotary-embedding.md)

```
+actor_rollout_ref.actor.megatron.override_transformer_config.position_embedding_type=rope \
+actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True \
```

3. 启动训练

训练脚本为`qw3-30bmoe-grpo-2node_base.sh`，配置训练脚本中所需的所有参数，然后在**所有节点**执行训练脚本。

```
# 确保在verl目录下执行训练脚本
cd verl

# 后台执行该训练脚本，并将日志保存
nohup bash qw3-30bmoe-grpo-2node_base.sh > qw3-30bmoe-grpo-2node_base.log 2>&1 &
tail -f qw3-30bmoe-grpo-2node_base.log
```

