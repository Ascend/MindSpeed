from mindspeed.te.fp8.recipes.block_scaling_recipe import BlockScalingRecipe
from mindspeed.te.fp8.recipes.delayed_scaling_recipe import DelayedScalingRecipe
from mindspeed.te.fp8.recipes.current_scaling_recipe import CurrentScalingRecipe
from mindspeed.te.fp8.recipes.mxfp8_scaling_recipe import MXFP8ScalingRecipe

SCALING_TYPE_MAP = {
    'delayed': DelayedScalingRecipe,
    'tensorwise': CurrentScalingRecipe,
    'blockwise': BlockScalingRecipe,
    'mxfp8': MXFP8ScalingRecipe,
}
