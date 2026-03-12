# Qwen3-8B with Megatron-Bridge

## 概述

本文给出一个使用 verl 完成 Qwen3-8B 训练（同时开启 Megatron-Bridge）的示例。

## 硬件环境

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 8 x Ascend NPUs |

注：本示例使用Atlas 800T A2单机进行训练

## 软件环境

| 组件 | 版本 |
|------|------|
| Python | 3.11 |
| CANN | 8.3.RC1 |
| torch | 2.7.1 |
| torch_npu | 2.7.1 |
| vLLM | 0.11.0 |
| vLLM-ascend | 0.11.0rc1 |
| Megatron-LM | v0.12.1 |

## 模型训练

1. 准备数据与权重  
   自行下载权重 [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/main)

   自行下载数据集 [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k/tree/main)

2. 配置训练脚本  
   更新脚本中的 `HF_MODEL_PATH`、`DIST_CKPT_PATH`、数据集路径与保存路径等参数。

3. 启动训练  

   ```
   # 确保在verl目录下执行训练脚本
   cd verl

   # 后台执行该训练脚本，并将日志保存
   nohup bash qwen3_8b/qw3-8b-grpo-1node_base.sh > qwen3_8b/qw3-8b-grpo-1node_base.log 2>&1 &
   tail -f qwen3_8b/qw3-8b-grpo-1node_base.log
   ```
