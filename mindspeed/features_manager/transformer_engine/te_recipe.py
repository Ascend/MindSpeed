# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""TeRecipeFeature: TransformerEngine NPU recipe enhancements.

This feature registers:
  - HiF8/MXFP4/W4A16 recipe choices extension on --fp8-format, --fp8-recipe, --fp4-recipe
  - HiF8 config CLI arguments (--hif8-input-margin, --hif8-weight-margin, etc.)
  - FP8/FP4 recipe wrapper patches (get_fp8_recipe, get_fp4_recipe)
  - HiF8 step recovery (train_step wrapper with NaN/Inf detection + retry)
"""

import logging

from mindspeed.features_manager.feature import MindSpeedFeature

logger = logging.getLogger("mindspeed.te_recipe")


class TeRecipeFeature(MindSpeedFeature):
    """TransformerEngine NPU recipe enhancements (HiF8, MXFP4, W4A16, step recovery)."""

    def __init__(self):
        super().__init__('te-recipe', optimization_level=0)

    def register_args(self, parser):
        # Extend fp8 / fp8_recipe choices for HiF8 support.
        self.add_parser_argument_choices_value(parser, "--fp8-format", "hif8")
        self.add_parser_argument_choices_value(parser, "--fp8-recipe", "hif8_delayed")
        # Extend fp4_recipe choices for MXFP4/W4A16 support.
        self.add_parser_argument_choices_value(parser, "--fp4-recipe", "mxfp4")
        self.add_parser_argument_choices_value(parser, "--fp4-recipe", "w4a16")

        group = parser.add_argument_group(title="te-recipe")
        group.add_argument(
            '--qat-scope',
            choices=['all', 'moe-only', 'linear-only', 'close'],
            default='all',
            help='Quantization scope: all modules, GroupedLinear only, Linear only, '
            'or close to disable all quantization for debugging.',
        )
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

        fp4_recipe_arg = getattr(args, "fp4_recipe", None)
        fp4_recipe = getattr(fp4_recipe_arg, "value", fp4_recipe_arg)
        if fp4_recipe == "w4a16":
            if getattr(args, "fp4", None) != "e2m1":
                raise ValueError("w4a16 recipe requires --fp4-format e2m1.")
            if getattr(args, "transformer_impl", None) != "transformer_engine":
                raise ValueError("w4a16 recipe requires --transformer-impl transformer_engine.")
            if getattr(args, "fp4_param_gather", False):
                raise ValueError("w4a16 recipe does not support --fp4-param-gather.")
            if getattr(args, "qat_scheme", None) is not None:
                raise ValueError("w4a16 recipe cannot be used together with --qat-scheme.")

    def register_patches(self, patch_manager, args):
        """TransformerEngine NPU patches: FP8/FP4 recipe wrappers + HiF8 step recovery."""
        try:
            from mindspeed.core.transformer_engine.transformer_engine import (
                HAVE_TE,
                core_transformer_config_from_args_wrapper,
                get_fp4_recipe_wrapper,
                get_fp8_recipe_wrapper,
            )

            patch_manager.register_patch(
                "megatron.training.arguments.core_transformer_config_from_args",
                core_transformer_config_from_args_wrapper,
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
