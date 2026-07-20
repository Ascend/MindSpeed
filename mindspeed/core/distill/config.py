# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Configuration for Quantization-Aware Distillation (QAD)."""

from dataclasses import dataclass, field


@dataclass
class QADConfig:
    """Configuration for Quantization-Aware Distillation.

    QAD uses a full-precision teacher model (precision handled by Megatron autocast)
    to guide a low-precision MXFP4 student model training via KL divergence loss,
    achieving accuracy recovery closer to BF16 baseline than QAT alone.

    """

    # Enable/disable QAD
    enabled: bool = False

    # Teacher model checkpoint path (MindSpeed-compatible format)
    teacher_checkpoint_path: str = ""

    # KL divergence temperature (T=1 for precise distribution matching)
    kl_temperature: float = 1.0

    # KL loss weight (alpha in total_loss = ce_loss + alpha * kl_loss)
    kl_loss_weight: float = 1.0

    # KL loss reduction method ("mean" or "sum")
    kl_reduction: str = "mean"

    # Additional custom fields (e.g., from yaml config)
    extra_config: dict = field(default_factory=dict)

    def validate(self):
        """Validate the QAD configuration.

        Returns:
            bool: True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.enabled:
            return True

        # Validate temperature
        if self.kl_temperature <= 0:
            raise ValueError(f"kl_temperature must be positive, got {self.kl_temperature}")

        # Validate loss weight
        if self.kl_loss_weight < 0:
            raise ValueError(f"kl_loss_weight must be non-negative, got {self.kl_loss_weight}")

        # Validate reduction method
        if self.kl_reduction not in ("mean", "sum"):
            raise ValueError(f"kl_reduction must be 'mean' or 'sum', got {self.kl_reduction}")

        # Validate teacher checkpoint path if provided
        if not self.teacher_checkpoint_path:
            raise ValueError("teacher_checkpoint_path is required when QAD is enabled")

        return True

    def total_loss_weight(self):
        """Get the total loss weight combination.

        QAD uses KL-only distillation. CE loss is for logging only and has
        zero gradient weight.

        Returns:
            tuple: (ce_loss_weight, kl_loss_weight) — ce_loss_weight is always 0.
        """
        return 0.0, self.kl_loss_weight
