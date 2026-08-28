# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""TeRecipeFeature: TransformerEngine NPU recipe enhancements.

This feature registers:
  - HiF8/MXFP4 recipe choices extension on --fp8-format, --fp8-recipe, --fp4-recipe
  - HiF8 config CLI arguments (--hif8-input-margin, --hif8-weight-margin, etc.)
  - FP8/FP4 recipe wrapper patches (get_fp8_recipe, get_fp4_recipe)
  - HiF8 step recovery (train_step wrapper with NaN/Inf detection + retry)
"""

import logging

from mindspeed.features_manager.feature import MindSpeedFeature

logger = logging.getLogger("mindspeed.te_recipe")


class TeRecipeFeature(MindSpeedFeature):
    """TransformerEngine NPU recipe enhancements (HiF8, MXFP4, step recovery)."""

    def __init__(self):
        super().__init__('te-recipe', optimization_level=0)

    def register_args(self, parser):
        # Extend fp8 / fp8_recipe choices for HiF8 support.
        self.add_parser_argument_choices_value(parser, "--fp8-format", "hif8")
        self.add_parser_argument_choices_value(parser, "--fp8-recipe", "hif8_delayed")
        # Extend fp4_recipe choices for MXFP4 support.
        self.add_parser_argument_choices_value(parser, "--fp4-recipe", "mxfp4")

        # HiF8 config fields. These are automatically injected onto
        # TransformerConfig by transformer_config_init_wrapper, so we only
        # need to register them with argparse here.
        group = parser.add_argument_group(title="te-recipe")
        group.add_argument(
            '--hif8-input-margin',
            type=int,
            default=11,
            help='Guard bits for input/activation tensors. Recommend range: 9-11.',
        )
        group.add_argument(
            '--hif8-weight-margin',
            type=int,
            default=12,
            help='Guard bits for weight tensors. Recommend range: 11-12.',
        )
        group.add_argument(
            '--hif8-grad-margin',
            type=int,
            default=11,
            help='Guard bits for gradient tensors. Recommend range: 9-11.',
        )
        group.add_argument(
            '--hif8-amax-collect-interval',
            type=int,
            default=5,
            help='Length of the warmup phase in iterations. During the first '
            'amax_collect_interval iterations the recipe uses current '
            'scaling and amax is collected every iteration. Recommend range: 5-20.',
        )
        group.add_argument(
            '--hif8-scale-update-interval',
            type=int,
            default=10,
            help='Number of iterations between amax history collections and scale factor updates in steady state.',
        )
        group.add_argument(
            '--hif8-amax-history-len',
            type=int,
            default=128,
            help='Length of the amax history buffer. Recommend choices: 64, 128, 256',
        )
        group.add_argument(
            '--no-hif8-step-recovery',
            action='store_true',
            default=False,
            help='Disable HiF8 NaN/Inf step recovery when using --fp8-recipe hif8_delayed. '
            'By default, step recovery is enabled when using hif8_delayed recipe.',
        )

    def validate_args(self, args):
        if args.fp8_recipe == 'hif8_delayed' and args.fp8 != 'hif8':
            raise ValueError("hif8_delayed recipe requires --fp8-format hif8.")

        if args.fp8 == 'hif8':
            if args.fp8_recipe not in ('tensorwise', 'delayed', 'hif8_delayed'):
                raise ValueError("hif8 only support tensorwise, delayed and hif8_delayed scaling type")

    def register_patches(self, patch_manager, args):
        """TransformerEngine NPU patches: FP8/FP4 recipe wrappers + HiF8 step recovery."""
        try:
            from mindspeed.core.transformer_engine.transformer_engine import (
                HAVE_TE,
                get_fp4_recipe_wrapper,
                get_fp8_recipe_wrapper,
            )

            if HAVE_TE:
                patch_manager.register_patch(
                    "megatron.core.fp8_utils.get_fp8_recipe",
                    get_fp8_recipe_wrapper,
                )
                patch_manager.register_patch(
                    "megatron.core.fp4_utils.get_fp4_recipe",
                    get_fp4_recipe_wrapper,
                )

            # HiF8 step recovery: wraps train_step with NaN/Inf detection + retry.
            # The wrapper itself short-circuits to the original train_step when
            # --fp8-recipe hif8_delayed is not used or --no-hif8-step-recovery is set,
            # so it is safe to register unconditionally.
            from mindspeed.core.transformer_engine.step_recovery.patch import (
                train_step_recovery_wrapper,
            )

            patch_manager.register_patch(
                "megatron.training.training.train_step",
                train_step_recovery_wrapper,
            )
            logger.debug("TransformerEngine NPU patches registered")
        except ImportError as e:
            logger.debug("TransformerEngine NPU patches skipped: %s", e)
