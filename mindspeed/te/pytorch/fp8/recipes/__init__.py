import torch

from mindspeed.args_utils import get_full_args as get_args
from .current_scaling_recipe import CurrentScalingRecipe, Float8CurrentScaling, TensorwiseMatMul
from .delayed_scaling_recipe import DelayedScalingRecipe, TEDelayedScaling
from .float8_block_scaling_recipe import Float8BlockRecipe, Float8BlockScaling, Float8BlockMatMul
from .mxfp8_scaling_recipe import MXFP8ScalingRecipe, MXFP8BlockScaling, MXFP8MatMul
from ..constants import Fp8Recipe

SCALING_TYPE_MAP = {
    Fp8Recipe.delayed: DelayedScalingRecipe,
    Fp8Recipe.tensorwise: CurrentScalingRecipe,
    Fp8Recipe.mxfp8: MXFP8ScalingRecipe,
    Fp8Recipe.blockwise: Float8BlockRecipe,
}

SCALING_CONFIG_MAP = {
    Fp8Recipe.delayed: TEDelayedScaling,
    Fp8Recipe.tensorwise: Float8CurrentScaling,
    Fp8Recipe.mxfp8: MXFP8BlockScaling,
    Fp8Recipe.blockwise: Float8BlockScaling,
}

MATMUL_MAP = {
    Fp8Recipe.mxfp8: MXFP8MatMul,
    Fp8Recipe.tensorwise: TensorwiseMatMul,
    Fp8Recipe.delayed: TensorwiseMatMul,
    Fp8Recipe.blockwise: Float8BlockMatMul,
}


def matmul_fp8(inputs, weight):
    if get_args().fp8_recipe not in MATMUL_MAP:
        return torch.matmul(inputs, weight.t())
    return MATMUL_MAP[get_args().fp8_recipe].apply(inputs, weight)
