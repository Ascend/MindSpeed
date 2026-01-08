import typing

INCLUDE_TYPEALIAS = {
    'FP8Metadata',
    'FP8Recipe',
    'FP8Tensor',
    'FP8RecipeScaling'
}

if typing.TYPE_CHECKING:
    from mindspeed.te.pytorch.fp8.metadata import FP8Metadata
    from mindspeed.te.pytorch.fp8.recipes import (
        MXFP8ScalingRecipe,
        CurrentScalingRecipe,
        DelayedScalingRecipe,
        BlockScalingRecipe,
        MXFP8BlockScaling,
        Float8CurrentScaling,
        TEDelayedScaling,
        BlockRecipeScaling
    )
    from mindspeed.te.pytorch.fp8 import Float8Tensor, Float8TensorWithTranspose, MXFP8Tensor

    FP8Recipe = typing.Union[CurrentScalingRecipe, DelayedScalingRecipe, BlockScalingRecipe, MXFP8ScalingRecipe]
    FP8RecipeScaling = typing.Union[Float8CurrentScaling, TEDelayedScaling, BlockRecipeScaling, MXFP8BlockScaling]
    FP8Tensor = typing.Union[Float8Tensor, Float8TensorWithTranspose, MXFP8Tensor]
else:

    def __getattr__(name):
        if name in INCLUDE_TYPEALIAS:
            return typing.TypeAlias
        raise AttributeError(f"module {__name__} has no attribute {name}")
