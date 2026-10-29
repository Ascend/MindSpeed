# QAD Quantization-Aware Distillation

## Background

After large language models are quantized to 4-bit (MXFP4/NVFP4), training can be accelerated, but the accuracy degrades significantly. Traditional QAT trains quantized models with cross entropy, which changes the output distribution and severely affects models trained with RL.

QAD (Quantization-Aware Distillation) uses a full-precision BF16 teacher model to guide the training of a low-precision student model through a KL-divergence loss, faithfully preserving the original distribution:

```text
L_total = α · KL(p_teacher ‖ p_student)
```

- **Teacher model**: BF16 full precision, with fixed parameters, providing a stable target distribution
- **Student model**: MXFP4 W4A4 quantization-aware training, fine-tunable
- **α**: KL loss weight (default 1.0), temperature parameter T=1
- **CE loss**: used only for log monitoring, not involved in gradient computation

## Core Components

| Component | Path | Description |
|------|------|------|
| QADConfig | `mindspeed/core/distill/config.py` | Configuration data class |
| TeacherModelManager | `mindspeed/core/distill/teacher_model_manager.py` | Teacher model lifecycle management |
| LogitsKLLoss | `mindspeed/core/distill/logits_kl_loss.py` | KL divergence loss computation |
| QADQuantEngineFeature | `mindspeed/features_manager/qad/qad_quant_engine.py` | Feature registration and parameter validation |
| QADForwardStepPatch | `mindspeed/core/distill/qad_adapter.py` | Training pipeline patch |

## Usage Constraints

> **Note:** QAD currently has the following limitations. Parameter validation is performed when QAD is enabled, and an error is reported to terminate the process if the conditions are not met:

| Limitation | Description |
|--------|------|
| **Dense models only** | MoE (Mixture of Experts) models are not supported. The teacher and student forward paths assume a single GPTModel logits output, which the MoE routing topology does not satisfy. If `--num-experts` is set when QAD is enabled, an error is reported. |
| **Pipeline parallelism (PP) not supported** | `pipeline-model-parallel-size > 1` is not supported. KL loss computation requires complete logits on every rank, but PP partitions layers across stages so that only the last stage produces logits. In addition, the batch cache replay logic assumes a single forward pass per micro-batch, which conflicts with multi-stage PP scheduling. Set `--pipeline-model-parallel-size 1`. |

## Usage

```bash
python pretrain_gpt.py \
    --qad-enable \
    --qad-teacher-load /path/to/bf16_teacher_checkpoint \
    --kl-temperature 1.0 \
    --kl-loss-weight 1.0 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    ... (other standard training parameters)
```

### Parameter Description

| Parameter | Default Value | Description |
|------|--------|------|
| `--qad-enable` | False | Enable QAD |
| `--qad-teacher-load` | "" | Path to the teacher model checkpoint (required when enabled) |
| `--kl-temperature` | 1.0 | KL divergence temperature parameter |
| `--kl-loss-weight` | 1.0 | KL loss weight α |
| `--kl-loss-reduction` | "mean" | KL loss reduction method (mean/sum) |
