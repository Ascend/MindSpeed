# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from argparse import ArgumentParser, Namespace

import pytest
from transformer_engine.common.recipe import W4A8BlockScaling

from mindspeed.core.transformer_engine.transformer_engine import (
    core_transformer_config_from_args_wrapper,
    get_fp4_recipe_wrapper,
)
from mindspeed.features_manager.transformer_engine.te_recipe import TeRecipeFeature


@pytest.mark.parametrize("scope", ["all", "moe-only", "linear-only", "close"])
@pytest.mark.parametrize("recipe_name", ["w4a16", "custom"])
def test_qat_scope_recipe_round_trip(scope, recipe_name):
    parser = ArgumentParser()
    TeRecipeFeature().register_args(parser)
    args = parser.parse_args(["--qat-scope", scope])
    args.fp8 = None
    args.fp8_recipe = "delayed"
    args.fp4_recipe = recipe_name
    args.fp4 = "e2m1"
    args.transformer_impl = "transformer_engine"
    TeRecipeFeature().validate_args(args)

    def build_config(args, *, config_class):
        return config_class(fp4_recipe=args.fp4_recipe)

    config = core_transformer_config_from_args_wrapper(build_config)(
        args, config_class=Namespace
    )
    original = W4A8BlockScaling(margin=2)
    recipe = get_fp4_recipe_wrapper(lambda config: original)(config)
    assert recipe.quantization_scope == scope
    if recipe_name == "custom":
        assert isinstance(recipe, W4A8BlockScaling)
        assert recipe.margin == 2
        assert original.quantization_scope == "all"


def test_qat_scope_defaults_and_invalid_choice():
    parser = ArgumentParser()
    TeRecipeFeature().register_args(parser)
    assert parser.parse_args([]).qat_scope == "all"
    with pytest.raises(SystemExit):
        parser.parse_args(["--qat-scope", "invalid"])
    config = Namespace(fp4_recipe="w4a16")
    assert get_fp4_recipe_wrapper(lambda config: None)(config).quantization_scope == "all"


@pytest.mark.parametrize("recipe_name", ["mxfp4", "custom"])
@pytest.mark.parametrize("scope", ["all", "moe-only", "linear-only", "close"])
def test_qat_scope_unsupported_recipe(scope, recipe_name):
    args = Namespace(fp8=None, fp8_recipe="delayed", fp4_recipe=recipe_name, qat_scope=scope)
    TeRecipeFeature().validate_args(args)
    config = Namespace(fp4_recipe=recipe_name, qat_scope=scope)
    get_recipe = get_fp4_recipe_wrapper(lambda config: Namespace())
    if scope == "all":
        assert get_recipe(config) is not None
    else:
        with pytest.raises(ValueError, match="does not support --qat-scope"):
            get_recipe(config)
