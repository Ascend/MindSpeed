# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause OR Apache-2.0 OR MIT

import warnings
from contextlib import nullcontext
from functools import wraps

from megatron.core.transformer.transformer_config import TransformerConfig


def quantize_param_shard(
    model_params,
    main_params,
    start_offsets,
    data_parallel_group,
    fsdp_shard_model_params=None,
):
    """Cast shard fp32 main params to fp8 model params."""

    warnings.warn("Currently, it is not supported to Cast shard fp32 main params to fp8 model params")


def get_fp8_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
    """Use Megatron's patched FP8/FP4 recipe factories for every MindSpeed path.

    The historical name also covers FP4. Duplicating recipe construction here
    bypasses HiF8, MXFP8-32x32, FP4 and quantization-scope extensions.
    """
    if not is_init and layer_no >= 0 and getattr(config, "first_last_layers_bf16", False):
        first = layer_no < config.num_layers_at_start_in_bf16
        last = layer_no >= config.num_layers - config.num_layers_at_end_in_bf16
        if first or last:
            # A nullcontext would inherit the outer FB-overlap block's FP8
            # state, quantizing layers explicitly configured to stay in BF16.
            from transformer_engine.pytorch import fp8_autocast

            return fp8_autocast(enabled=False)
    if getattr(config, "fp8", None) or (is_init and getattr(config, "fp8_param", False)):
        from megatron.core.fp8_utils import get_fp8_context as megatron_fp8_context

        return megatron_fp8_context(config, layer_no, is_init)
    if getattr(config, "fp4", None) or (is_init and getattr(config, "fp4_param", False)):
        from megatron.core.fp4_utils import get_fp4_context

        return get_fp4_context(config, layer_no, is_init)
    return nullcontext()


def fp8_context_wrapper(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
    """Wraps the fp8_context_wrapper function."""

    def wrapper_fn(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with get_fp8_context(config, layer_no, is_init):
                return fn(*args, **kwargs)

        return wrapper

    return wrapper_fn
