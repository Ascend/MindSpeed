# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import importlib.util
from argparse import ArgumentParser, Namespace
from functools import wraps

from mindspeed.features_manager.feature import MindSpeedFeature


def transformer_config_batch_invariant_mode_wrapper(fn):
    @wraps(fn)
    def wrapper(self):
        use_flash_attn_npu_batch_invariant = getattr(self, "use_flash_attn_npu_batch_invariant", False)
        batch_invariant_mode = getattr(self, "batch_invariant_mode", False)
        if not (use_flash_attn_npu_batch_invariant and batch_invariant_mode):
            return fn(self)

        self.batch_invariant_mode = False
        try:
            return fn(self)
        finally:
            self.batch_invariant_mode = batch_invariant_mode

    return wrapper


class BatchInvariantFeature(MindSpeedFeature):
    """Patch batch-invariant kernels and related attention backend."""

    def __init__(self):
        super().__init__("use-batch-invariant-ops", optimization_level=2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title="batch invariant")
        group.add_argument(
            "--use-batch-invariant-ops",
            action="store_true",
            default=False,
            help="Use batch invariant operators for train/inference consistency.",
        )
        group.add_argument(
            "--use-flash-attn-npu-batch-invariant",
            action="store_true",
            default=False,
            help="Use flash-attention-npu attention in batch invariant mode.",
        )

    def is_need_apply(self, args):
        return (
            getattr(args, "use_batch_invariant_ops", False)
            or getattr(args, "use_flash_attn_npu_batch_invariant", False)
            or getattr(args, "batch_invariant_mode", False)
        )

    def validate_args(self, args: Namespace):
        if not getattr(args, "use_flash_attn_npu_batch_invariant", False):
            return

        if getattr(args, "tp_2d", False):
            raise AssertionError("--use-flash-attn-npu-batch-invariant does not support 2D tensor parallelism yet.")
        if int(getattr(args, "context_parallel_size", 1)) != 1:
            raise AssertionError("--use-flash-attn-npu-batch-invariant does not support context parallelism yet.")

        # Dependency check
        if importlib.util.find_spec("flash_attn_npu") is None:
            raise ImportError("--use-flash-attn-npu-batch-invariant requires flash-attention-npu to be installed.")

    def register_patches(self, patch_manager, args: Namespace):
        if getattr(args, "use_flash_attn_npu_batch_invariant", False):
            from mindspeed.core.transformer.flash_attention.flash_attn_npu.adaptor import (
                dot_product_attention_flash_attn_npu_forward,
            )

            patch_manager.register_patch(
                "megatron.core.transformer.dot_product_attention.DotProductAttention.forward",
                dot_product_attention_flash_attn_npu_forward,
            )

        if (
            getattr(args, "use_flash_attn_npu_batch_invariant", False)
            and getattr(args, "batch_invariant_mode", False)
        ):
            patch_manager.register_patch(
                "megatron.core.transformer.transformer_config.TransformerConfig.__post_init__",
                transformer_config_batch_invariant_mode_wrapper,
            )
