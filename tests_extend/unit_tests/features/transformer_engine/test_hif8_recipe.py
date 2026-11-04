# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from argparse import Namespace

import pytest

te = pytest.importorskip("transformer_engine")
from transformer_engine.common import recipe as te_recipe  # noqa: E402

from mindspeed.core.transformer_engine.transformer_engine import (  # noqa: E402
    core_transformer_config_from_args_wrapper,
    get_fp8_recipe_wrapper,
)


def _hif8_recipe_cls():
    recipe_cls = getattr(te_recipe, "HIF8DelayedScaling", None)
    if recipe_cls is None:
        pytest.skip("This Transformer Engine build has no HIF8DelayedScaling")
    return recipe_cls


def _config(**overrides):
    values = dict(
        fp8="hif8",
        fp8_recipe="delayed",
        fp8_wgrad=True,
        fp8_amax_compute_algo="max",
        fp8_dot_product_attention=False,
        hif8_amax_history_len=128,
        hif8_input_margin=11,
        hif8_weight_margin=12,
        hif8_grad_margin=11,
        hif8_amax_collect_interval=5,
        hif8_scale_update_interval=10,
        qat_scope="all",
    )
    values.update(overrides)
    return Namespace(**values)


def test_hif8_format_with_delayed_recipe_uses_hif8_delayed_scaling():
    recipe = get_fp8_recipe_wrapper(lambda config: None)(_config())

    assert isinstance(recipe, _hif8_recipe_cls())
    assert recipe.fp8_format.name.startswith("HIF8")
    assert recipe.amax_collect_interval == 5
    assert recipe.scale_update_interval == 10
    assert recipe.amax_history_len == 128
    assert recipe.hif8_input_margin == 11
    assert recipe.hif8_weight_margin == 12
    assert recipe.hif8_grad_margin == 11


def test_non_hif8_format_falls_through_to_megatron_recipe():
    sentinel = object()

    def original(config):
        return sentinel

    recipe = get_fp8_recipe_wrapper(original)(_config(fp8="e4m3"))

    assert recipe is sentinel


def test_hif8_cli_args_are_propagated_into_config():
    args = Namespace(
        qat_scope="all",
        hif8_amax_history_len=32,
        hif8_input_margin=9,
        hif8_weight_margin=10,
        hif8_grad_margin=9,
        hif8_amax_collect_interval=3,
        hif8_scale_update_interval=7,
    )
    config = core_transformer_config_from_args_wrapper(lambda _: Namespace())(args)
    config.fp8 = "hif8"
    config.fp8_recipe = "delayed"
    config.fp8_wgrad = True
    config.fp8_amax_compute_algo = "max"

    recipe = get_fp8_recipe_wrapper(lambda _: None)(config)

    assert recipe.amax_history_len == 32
    assert recipe.hif8_input_margin == 9
    assert recipe.hif8_weight_margin == 10
    assert recipe.hif8_grad_margin == 9
    assert recipe.amax_collect_interval == 3
    assert recipe.scale_update_interval == 7


def test_hif8_recipe_rejects_non_default_qat_scope():
    with pytest.raises(ValueError, match="qat-scope"):
        get_fp8_recipe_wrapper(lambda _: None)(_config(qat_scope="moe-only"))
