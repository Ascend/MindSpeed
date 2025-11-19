import torch

from mindspeed.args_utils import get_full_args as get_args
from .block_scaling_recipe import BlockScalingRecipe, BlockRecipeScaling, BlockwiseMatMul
from .current_scaling_recipe import CurrentScalingRecipe, Float8CurrentScaling, TensorwiseMatMul
from .delayed_scaling_recipe import DelayedScalingRecipe, TEDelayedScaling
from .groupwise_scaling_recipe import GroupwiseScalingRecipe, GroupwiseBlockScaling, GroupwiseMatMul
from .mxfp8_scaling_recipe import MXFP8ScalingRecipe, MXFP8BlockScaling, MXFP8MatMul
from ..constants import Fp8Recipe

SCALING_TYPE_MAP = {
    Fp8Recipe.delayed: DelayedScalingRecipe,
    Fp8Recipe.tensorwise: CurrentScalingRecipe,
    Fp8Recipe.blockwise: BlockScalingRecipe,
    Fp8Recipe.mxfp8: MXFP8ScalingRecipe,
    Fp8Recipe.groupwise: GroupwiseScalingRecipe,
}

SCALING_CONFIG_MAP = {
    Fp8Recipe.delayed: TEDelayedScaling,
    Fp8Recipe.tensorwise: Float8CurrentScaling,
    Fp8Recipe.blockwise: BlockRecipeScaling,
    Fp8Recipe.mxfp8: MXFP8BlockScaling,
    Fp8Recipe.groupwise: GroupwiseBlockScaling,
}

MATMUL_MAP = {
    Fp8Recipe.mxfp8: MXFP8MatMul,
    Fp8Recipe.tensorwise: TensorwiseMatMul,
    Fp8Recipe.delayed: TensorwiseMatMul,
    Fp8Recipe.groupwise: GroupwiseMatMul,
    Fp8Recipe.blockwise: BlockwiseMatMul,
}


def matmul_fp8(inputs, weight):
    if get_args().fp8_recipe not in MATMUL_MAP:
        return torch.matmul(inputs, weight.t())
    return MATMUL_MAP[get_args().fp8_recipe].apply(inputs, weight)
