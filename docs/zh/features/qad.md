# QAD 量化感知蒸馏

## 背景

大语言模型量化到4-bit（MXFP4/NVFP4）后可获得训练加速，但精度显著退化。传统QAT使用交叉熵训练量化模型，会改变输出分布，对RL训练模型影响严重。

QAD（Quantization-Aware Distillation）使用全精度BF16教师模型通过KL散度损失引导低精度学生模型训练，忠实保留原始分布：

```text
L_total = α · KL(p_teacher ‖ p_student)
```

- **教师模型**：BF16全精度，固定参数，提供稳定目标分布
- **学生模型**：MXFP4 W4A4量化感知训练，可微调
- **α**：KL损失权重（默认1.0），温度参数T=1
- **CE损失**：仅用于日志监控，不参与梯度计算

## 核心组件

| 组件 | 路径 | 说明 |
|------|------|------|
| QADConfig | `mindspeed/core/distill/config.py` | 配置数据类 |
| TeacherModelManager | `mindspeed/core/distill/teacher_model_manager.py` | 教师模型生命周期管理 |
| LogitsKLLoss | `mindspeed/core/distill/logits_kl_loss.py` | KL散度损失计算 |
| QADQuantEngineFeature | `mindspeed/features_manager/qad/qad_quant_engine.py` | 特性注册与参数校验 |
| QADForwardStepPatch | `mindspeed/core/distill/qad_adapter.py` | 训练流水线补丁 |

## 使用限制

> **注意：** QAD 当前存在以下限制，启用时会进行参数校验，不满足条件将报错终止：

| 限制项 | 说明 |
|--------|------|
| **仅支持 Dense 模型** | 不支持 MoE（Mixture of Experts）模型。教师与学生前向路径假设单一 GPTModel logits 输出，MoE 路由拓扑不满足此假设。启用 QAD 时若设置了 `--num-experts` 将报错。 |
| **不支持流水线并行（PP）** | 不支持 `pipeline-model-parallel-size > 1`。KL 损失计算要求每个 rank 上有完整 logits，但 PP 将层划分到不同 stage，仅最后一个 stage 产生 logits。此外，批次缓存重放逻辑假设每个 micro-batch 单次前向，与 PP 多 stage 调度冲突。请设置 `--pipeline-model-parallel-size 1`。 |

## 使用方法

```bash
python pretrain_gpt.py \
    --qad-enable \
    --qad-teacher-load /path/to/bf16_teacher_checkpoint \
    --kl-temperature 1.0 \
    --kl-loss-weight 1.0 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    ... (其他标准训练参数)
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--qad-enable` | False | 启用QAD |
| `--qad-teacher-load` | "" | 教师模型检查点路径（启用时必填） |
| `--kl-temperature` | 1.0 | KL散度温度参数 |
| `--kl-loss-weight` | 1.0 | KL损失权重α |
| `--kl-loss-reduction` | "mean" | KL损失归约方式（mean/sum） |
