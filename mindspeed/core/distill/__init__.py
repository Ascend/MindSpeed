# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Core distillation modules for Quantization-Aware Distillation (QAD)."""

from mindspeed.core.distill.config import QADConfig
from mindspeed.core.distill.teacher_model_manager import TeacherModelManager
from mindspeed.core.distill.logits_kl_loss import LogitsKLLoss

__all__ = [
    "QADConfig",
    "TeacherModelManager",
    "LogitsKLLoss",
]
