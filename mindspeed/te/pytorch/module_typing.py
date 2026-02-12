import typing

INCLUDE_TYPEALIAS = {
    'FP8Metadata',
    'FP8Recipe',
    'FP8Tensor',
    'FP8RecipeScaling'
}

if typing.TYPE_CHECKING:
    import torch

    # noinspection PyUnusedImports
    from mindspeed.te.pytorch.fp8.metadata import FP8Metadata
    from mindspeed.te.pytorch.fp8.recipes import (
        MXFP8ScalingRecipe,
        CurrentScalingRecipe,
        DelayedScalingRecipe,
        Float8BlockRecipe,
        MXFP8BlockScaling,
        Float8CurrentScaling,
        TEDelayedScaling,
        Float8BlockScaling,
    )
    from mindspeed.te.pytorch.fp8.tensor import Float8Tensor, Float8Tensor2D, MXFP8Tensor, Float8BlockTensor

    FP8Recipe = typing.Union[CurrentScalingRecipe, DelayedScalingRecipe, Float8BlockRecipe, MXFP8ScalingRecipe]
    FP8RecipeScaling = typing.Union[Float8CurrentScaling, TEDelayedScaling, Float8BlockScaling, MXFP8BlockScaling]
    FP8Tensor = typing.Union[Float8Tensor, Float8Tensor2D, Float8BlockTensor, MXFP8Tensor, torch.Tenror]
else:

    def __getattr__(name):
        if name in INCLUDE_TYPEALIAS:
            return typing.TypeAlias
        raise AttributeError(f"module {__name__} has no attribute {name}")
