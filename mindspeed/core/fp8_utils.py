# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause OR Apache-2.0 OR MIT

import warnings
from contextlib import nullcontext
from functools import wraps

from megatron.core import parallel_state
from megatron.core.enums import Fp8Recipe
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.extensions.transformer_engine import TEDelayedScaling


def quantize_param_shard(
    model_params, main_params, start_offsets, data_parallel_group, fsdp_shard_model_params=None
):
    """Cast shard fp32 main params to fp8 model params."""

    warnings.warn("Currently, it is not supported to Cast shard fp32 main params to fp8 model params")


def get_fp8_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
    """Return fp8 context manager.

    Arguments:
        config (TransformerConfig): Configuration object.
        layer_no (int): *Global* layer index (including layers on other
            pipeline-parallel ranks).
        is_init (bool): Whether the context is fp8_model_init (True) or fp8_autocast (False).

    Returns:
        FP8 context.
        If layer_no < 0, we return a fp8 context for all layers regardless of layer_no.
        We return nullcontext() when: a) not using fp8 to train, b) layer_no is a layer
        that needs to be trained in bf16.
    """
    num_bf16_layers_at_start = (
        config.num_layers_at_start_in_bf16 if config.first_last_layers_bf16 else 0
    )
    num_bf16_layers_at_end = (
        config.num_layers_at_end_in_bf16 if config.first_last_layers_bf16 else 0
    )
    # Since layer_no is a global layer index, additional checks on whether
    # we are in the first or last pipeline-parallel rank are not needed.
    is_first_layer = layer_no < num_bf16_layers_at_start
    is_last_layer = layer_no >= config.num_layers - num_bf16_layers_at_end

    need_fp8_context = config.fp8 if not is_init else config.fp8_param

    if not need_fp8_context:
        # bf16 training
        fp8_context = nullcontext()
    elif layer_no >= 0 and config.first_last_layers_bf16 and (is_first_layer or is_last_layer):
        # fp8 training but this layer_no should be bf16
        fp8_context = nullcontext()
    else:
        from transformer_engine.common.recipe import Float8CurrentScaling, MXFP8BlockScaling, Format
        from transformer_engine.pytorch import fp8_autocast, fp8_model_init
        from mindspeed.te.pytorch.fp8.recipes import BlockRecipeScaling, GroupwiseBlockScaling
        if config.fp8 == "e4m3":
            fp8_format = Format.E4M3
        elif config.fp8 == "hybrid":
            fp8_format = Format.HYBRID
        elif config.fp8 == 'hif8':
            fp8_format = Format.HIF8
        else:
            raise ValueError("E4M3, HYBRID and hif8 are the only supported FP8 formats.")

        # Select fp8 recipe (TE version >= 2.1.0).
        if config.fp8_recipe == Fp8Recipe.delayed:
            fp8_recipe = TEDelayedScaling(
                config=config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not config.fp8_wgrad),
            )
        elif config.fp8_recipe == Fp8Recipe.tensorwise:
            fp8_recipe = Float8CurrentScaling(
                fp8_format=fp8_format
            )
        elif config.fp8_recipe == Fp8Recipe.mxfp8:
            fp8_recipe = MXFP8BlockScaling(
                fp8_format=fp8_format
            )
        elif config.fp8_recipe == Fp8Recipe.blockwise:
            fp8_recipe = BlockRecipeScaling(
                fp8_format=fp8_format
            )
        elif config.fp8_recipe == Fp8Recipe.groupwise:
            fp8_recipe = GroupwiseBlockScaling(
                fp8_format=fp8_format
            )
        else:
            raise ValueError(
                "Float8CurrentScaling, MXFP8BlockScaling and DelayedScaling are "
                "the only supported FP8 recipes."
            )
        fp8_group = None
        if parallel_state.model_parallel_is_initialized():
            fp8_group = parallel_state.get_amax_reduction_group(
                with_context_parallel=True, tp_only_amax_red=config.tp_only_amax_red
            )

        if not is_init:
            fp8_context = fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )
        else:
            import inspect

            context_args = {"enabled": True}
            # Check if fp8_model_init supports setting recipe
            if "recipe" in (
                inspect.signature(fp8_model_init).parameters
            ):
                context_args["recipe"] = fp8_recipe
            # Check if fp8_model_init supports preserve_high_precision_init_val
            if "preserve_high_precision_init_val" in (
                inspect.signature(fp8_model_init).parameters
            ):
                context_args["preserve_high_precision_init_val"] = True
            fp8_context = fp8_model_init(**context_args)

        # First / last layer in bf16 isn't supported with delayed scaling since it
        # requires entering/exiting fp8 context per layer, causing incorrect amax
        # reduction behavior.
        assert not (
            config.first_last_layers_bf16 and isinstance(fp8_recipe, TEDelayedScaling)
        ), "Delayed scaling does not support first / last layer in BF16."

    return fp8_context


def fp8_context_wrapper(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
    """Wraps the fp8_context_wrapper function."""

    def wrapper_fn(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with get_fp8_context(config, layer_no, is_init):
                return fn(*args, **kwargs)

        return wrapper

    return wrapper_fn
