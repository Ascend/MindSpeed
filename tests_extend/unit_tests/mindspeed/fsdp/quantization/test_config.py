import pytest

from mindspeed.fsdp.quantization.config import (
    ScalingGranularity,
    ScalingGranularityEnum,
    ScalingStrategyEnum,
    QuantRecipe,
    _registry_quant_recipes,
    _dtype_mapping,
)


class TestScalingGranularity:
    @pytest.mark.parametrize("block_size", [[1, 1, 32], [1, 32, 32]])
    def test_valid_block_size(self, block_size):
        granularity = ScalingGranularity(ScalingGranularityEnum.MX, block_size)
        assert granularity.stype == ScalingGranularityEnum.MX
        assert granularity.block_size == block_size

    @pytest.mark.parametrize("block_size", [[1, 1, 16], [2, 2, 32], None])
    def test_invalid_block_size_raises_error(self, block_size):
        with pytest.raises(ValueError) as exc_info:
            ScalingGranularity(ScalingGranularityEnum.MX, block_size)
        assert "Invalid block size" in str(exc_info.value)

    def test_per_tensor_with_none_block_size(self):
        granularity = ScalingGranularity(ScalingGranularityEnum.PER_TENSOR, None)
        assert granularity.stype == ScalingGranularityEnum.PER_TENSOR
        assert granularity.block_size is None


class TestMxFP8Recipe:
    @pytest.mark.parametrize(
        "recipe_name,expected_block_size",
        [
            ("mxfp8", [1, 1, 32]),
            ("mxfp8-32x32", [1, 32, 32]),
        ],
    )
    def test_mxfp8_recipe_registered(self, recipe_name, expected_block_size):
        assert recipe_name in _registry_quant_recipes
        recipe_func = _registry_quant_recipes[recipe_name]
        recipe = recipe_func()

        assert recipe.scaling_strategy == ScalingStrategyEnum.DYNAMIC
        assert recipe.scaling_granularity.stype == ScalingGranularityEnum.MX
        assert recipe.scaling_granularity.block_size == expected_block_size
        assert recipe.inputs_dtype == _dtype_mapping["E4M3"]
        assert recipe.weight_dtype == _dtype_mapping["E4M3"]
        assert recipe.grads_dtype == _dtype_mapping["E4M3"]

    def test_mxfp8_and_mxfp8_32x32_have_different_block_sizes(self):
        mxfp8_recipe = _registry_quant_recipes["mxfp8"]()
        mxfp8_32x32_recipe = _registry_quant_recipes["mxfp8-32x32"]()

        assert mxfp8_recipe.scaling_granularity.block_size == [1, 1, 32]
        assert mxfp8_32x32_recipe.scaling_granularity.block_size == [1, 32, 32]
        assert mxfp8_recipe.scaling_granularity.block_size != mxfp8_32x32_recipe.scaling_granularity.block_size


class TestQuantRecipeFromName:
    @pytest.mark.parametrize(
        "recipe_name,expected_block_size",
        [
            ("mxfp8", [1, 1, 32]),
            ("mxfp8-32x32", [1, 32, 32]),
        ],
    )
    def test_from_recipe_name(self, recipe_name, expected_block_size):
        recipe_func = QuantRecipe.from_recipe_name(recipe_name)
        recipe = recipe_func()
        assert recipe.scaling_strategy == ScalingStrategyEnum.DYNAMIC
        assert recipe.scaling_granularity.stype == ScalingGranularityEnum.MX
        assert recipe.scaling_granularity.block_size == expected_block_size

    def test_from_recipe_name_invalid_raises_error(self):
        with pytest.raises(ValueError) as exc_info:
            QuantRecipe.from_recipe_name("invalid_recipe")
        assert "Unknown recipe name" in str(exc_info.value)


class TestQuantRecipeGetKeyDtype:
    @pytest.mark.parametrize("key", ["inputs", "weight", "grads"])
    def test_get_key_dtype(self, key):
        recipe_func = QuantRecipe.from_recipe_name("mxfp8-32x32")
        recipe = recipe_func()
        dtype = recipe.get_key_dtype(key)
        assert dtype == _dtype_mapping["E4M3"]

    def test_get_key_dtype_invalid_key_raises_error(self):
        recipe_func = QuantRecipe.from_recipe_name("mxfp8-32x32")
        recipe = recipe_func()
        with pytest.raises(ValueError) as exc_info:
            recipe.get_key_dtype("invalid_key")
        assert "Unknown key" in str(exc_info.value)
