# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""TransformerEngine NPU patches.

Provides NPU-specific wrappers for TransformerEngine-related functions:
  - FP8 recipe with HiF8 format support
  - FP4 recipes with MXFP4 and W4A16 support
"""

from dataclasses import replace
from functools import wraps

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.extensions.transformer_engine import TEDelayedScaling

HAVE_TE = False
try:
    import transformer_engine  # pylint: disable=W0611

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    pass


def get_fp8_recipe_wrapper(fn):
    """Wrap get_fp8_recipe to support MindSpeed/TENPU recipe extensions."""

    @wraps(fn)
    def wrapper(config: TransformerConfig):
        fp8_recipe = getattr(config.fp8_recipe, "value", config.fp8_recipe)
        if fp8_recipe == "mxfp8":
            from mindspeed.args_utils import get_full_args

            if getattr(get_full_args(), "mxfp8_32x32", False):
                try:
                    from transformer_engine.common.recipe import MXFP832x32BlockScaling
                except ImportError as exc:
                    raise RuntimeError("mxfp8-32x32 requires the matching TENPU recipe implementation.") from exc

                if config.fp8 == "e4m3":
                    fp8_format = transformer_engine.common.recipe.Format.E4M3
                elif config.fp8 == "hybrid":
                    fp8_format = transformer_engine.common.recipe.Format.HYBRID
                else:
                    raise ValueError("mxfp8-32x32 supports E4M3 and HYBRID FP8 formats only.")
                return MXFP832x32BlockScaling(fp8_format=fp8_format)

        if config.fp8 == "hif8":
            fp8_format = transformer_engine.common.recipe.Format.HIF8
            if fp8_recipe == "delayed":
                return TEDelayedScaling(
                    config=config,
                    fp8_format=fp8_format,
                    override_linear_precision=(False, False, not config.fp8_wgrad),
                )
            if fp8_recipe == "tensorwise":
                return transformer_engine.common.recipe.Float8CurrentScaling(
                    fp8_format=fp8_format, fp8_dpa=config.fp8_dot_product_attention
                )
            if fp8_recipe == "hif8_delayed":
                return transformer_engine.common.recipe.HIF8DelayedScaling(
                    fp8_format=fp8_format,
                    amax_compute_algo=config.fp8_amax_compute_algo,
                    amax_history_len=config.hif8_amax_history_len,
                    override_linear_precision=(False, False, not config.fp8_wgrad),
                    hif8_input_margin=config.hif8_input_margin,
                    hif8_weight_margin=config.hif8_weight_margin,
                    hif8_grad_margin=config.hif8_grad_margin,
                    amax_collect_interval=config.hif8_amax_collect_interval,
                    scale_update_interval=config.hif8_scale_update_interval,
                )
            raise ValueError(
                "DelayedScaling, Float8CurrentScaling and HIF8DelayedScaling are the only supported HIF8 recipes."
            )

        return fn(config)

    return wrapper


def core_transformer_config_from_args_wrapper(fn):
    """Propagate the MindSpeed QAT scope into the model configuration."""

    @wraps(fn)
    def wrapper(args, *pos_args, **kwargs):
        config = fn(args, *pos_args, **kwargs)
        config.qat_scope = getattr(args, "qat_scope", "all")
        return config

    return wrapper


def get_fp4_recipe_wrapper(fn):
    """Wrap get_fp4_recipe to support NPU FP4 recipes."""

    @wraps(fn)
    def wrapper(config):
        fp4_recipe = getattr(config.fp4_recipe, "value", config.fp4_recipe)
        recipe_cls_name = {
            "mxfp4": "MXFP4BlockScaling",
            "w4a16": "W4A16BlockScaling",
        }.get(fp4_recipe)
        recipes = transformer_engine.common.recipe
        scope = getattr(config, "qat_scope", "all")
        if recipe_cls_name is not None:
            recipe = getattr(recipes, recipe_cls_name)()
        else:
            recipe = fn(config)
        if isinstance(recipe, recipes.ScopedRecipe):
            return replace(recipe, quantization_scope=scope)
        if scope != "all":
            raise ValueError(f"{fp4_recipe} does not support --qat-scope.")
        return recipe

    return wrapper
