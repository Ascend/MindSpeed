# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""QAD (Quantization-Aware Distillation) feature manager.

Integrates QAD training into the MindSpeed training pipeline following
the MindSpeedFeature interface pattern.
"""

import argparse
import logging

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager

logger = logging.getLogger(__name__)


class QADQuantEngineFeature(MindSpeedFeature):
    """Feature manager for Quantization-Aware Distillation (QAD).

    QAD uses a full-precision teacher model (precision handled by Megatron autocast)
    to guide a low-precision MXFP4 student model training via KL divergence loss.

    This feature:
    1. Registers QAD command-line arguments
    2. Patches model_provider to create teacher + student models
    3. Patches forward_step to run both teacher and student forward passes
    4. Patches loss_func to compute KL-only loss (CE for logging only)
    """

    def __init__(self):
        """Initialize QADQuantEngineFeature."""
        super().__init__('qad-quant-engine', optimization_level=0)
        logger.debug("[QAD] QADQuantEngineFeature initialized (mindspeed base)")

    def is_need_apply(self, args):
        """Only apply QAD patches when --qad-enable is explicitly set.

        Override the default behavior (optimization_level=0 means always apply)
        to avoid importing mindspeed_llm when running MindSpeed-MM.
        """
        return getattr(args, 'qad_enable', False)

    def register_args(self, parser: argparse.ArgumentParser):
        """Register QAD command-line arguments.

        Args:
            parser: ArgumentParser instance
        """
        group = parser.add_argument_group(title='qad')

        # Enable/disable QAD
        group.add_argument(
            '--qad-enable', action='store_true', default=False, help='Enable Quantization-Aware Distillation (QAD)'
        )
        group.add_argument('--no-qad-enable', dest='qad_enable', action='store_false', help='Disable QAD')

        # Teacher model checkpoint path
        group.add_argument(
            '--qad-teacher-load', type=str, default='', help='Path to the teacher model checkpoint directory'
        )

        # KL loss configuration
        group.add_argument(
            '--kl-temperature', type=float, default=1.0, help='Temperature for KL divergence loss (default: 1.0)'
        )
        group.add_argument(
            '--kl-loss-weight', type=float, default=1.0, help='Weight for KL loss in total loss (default: 1.0)'
        )
        group.add_argument(
            '--kl-loss-reduction',
            type=str,
            default='mean',
            choices=['mean', 'sum'],
            help='KL loss reduction method (default: mean)',
        )

        # QAD uses KL-only distillation. CE loss is computed for logging only
        # and does not contribute to gradients. This avoids the double student
        # forward pass and num_tokens scaling complexity of CE+KL mode.

        # Teacher freezing is mandatory in QAD (a mathematical requirement of
        # knowledge distillation) and is always applied unconditionally in
        # TeacherModelManager.load_teacher().

    def pre_validate_args(self, args):
        """Pre-validate QAD arguments before main validation.

        Raises AssertionError on invalid args to fail fast, matching the
        convention used by other MindSpeedFeature subclasses
        (e.g. FusedEmaAdamwFeature.pre_validate_args).

        Args:
            args: Parsed arguments

        Raises:
            AssertionError: If any QAD argument is invalid.
        """
        logger.debug("[QAD] pre_validate_args called, qad_enable=%s", getattr(args, 'qad_enable', False))
        if not getattr(args, 'qad_enable', False):
            return

        # Check teacher checkpoint path
        teacher_path = getattr(args, 'qad_teacher_load', '')
        if not teacher_path:
            raise AssertionError("--qad-teacher-load is required when QAD is enabled (--qad-enable).")

        # Validate KL temperature
        kl_temp = getattr(args, 'kl_temperature', 1.0)
        if kl_temp <= 0:
            raise AssertionError(f"--kl-temperature must be positive, got {kl_temp}.")

        # Validate KL loss weight
        kl_weight = getattr(args, 'kl_loss_weight', 1.0)
        if kl_weight < 0:
            raise AssertionError(f"--kl-loss-weight must be non-negative, got {kl_weight}.")

        # Validate KL reduction
        kl_reduction = getattr(args, 'kl_loss_reduction', 'mean')
        if kl_reduction not in ('mean', 'sum'):
            raise AssertionError(f"--kl-loss-reduction must be 'mean' or 'sum', got '{kl_reduction}'.")

        # QAD currently only supports dense (non-MoE) models. The teacher and
        # student forward paths assume a single GPTModel logits output, which
        # does not hold for MoE routing topologies.
        num_experts = getattr(args, 'num_experts', None)
        if num_experts:
            raise AssertionError(
                "QAD does not support MoE models (--num-experts is set). Only dense models are supported."
            )

        # QAD does not support pipeline parallelism. The KL loss computation
        # requires full logits on every rank, but PP partitions layers across
        # stages so only the last stage produces logits. Additionally, the
        # _BatchCacheIterator replay logic assumes a single forward pass per
        # micro-batch, which conflicts with PP's multi-stage schedule.
        pp_size = getattr(args, 'pipeline_model_parallel_size', 1)
        if pp_size > 1:
            raise AssertionError(
                f"QAD does not support pipeline parallelism "
                f"(pipeline-model-parallel-size={pp_size}). Please set "
                f"--pipeline-model-parallel-size to 1."
            )

    def register_patches(self, patch_manager: MindSpeedPatchesManager, args):
        """Register patches for QAD integration.

        Args:
            patch_manager: MindSpeedPatchesManager instance
            args: Parsed arguments
        """
        logger.debug("[QAD] register_patches called, qad_enable=%s", getattr(args, 'qad_enable', False))
        if not getattr(args, 'qad_enable', False):
            logger.debug("[QAD] QAD not enabled, skipping patch registration")
            return

        # Register QAD patches at megatron level (clean dependency direction:
        # MindSpeed patches megatron.training.training.* functions that every
        # pretrain path — including MindSpeed-LLM's own reimplementation — calls,
        # instead of patching a consumer app's pretrain directly).
        #   - setup_model_and_optimizer: injects qad_model_provider (teacher loading)
        #   - train_step: injects qad_forward_step (KL distillation loss)
        from mindspeed.core.distill.qad_adapter import (
            QADForwardStepPatch,
        )

        patch_manager.register_patch(
            'megatron.training.training.setup_model_and_optimizer',
            QADForwardStepPatch.patched_setup_model_and_optimizer_wrapper,
        )
        patch_manager.register_patch(
            'megatron.training.training.train_step', QADForwardStepPatch.patched_train_step_wrapper
        )
        logger.debug("[QAD] Registered setup_model_and_optimizer and train_step patches")

        logger.info("[QAD] Patches registered successfully")
