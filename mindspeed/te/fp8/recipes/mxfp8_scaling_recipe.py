import torch

from mindspeed.te.fp8.recipes.recipe import Recipe, RecipeConfig
from mindspeed.te.fp8.scale_data import ScaleData


class MXFP8ScalingRecipe(Recipe):

    def __init__(self, key, recipe_config: RecipeConfig, tensor_shape) -> None:
        super().__init__(key, recipe_config, tensor_shape)
        self.block_dim = None
        self.scale = ScaleData(recipe_config)

    def quantization(self, tensor, scale_data: ScaleData):
        if tensor is None:
            return tensor
        new_tensor = scale_data.mxfp8_quantization(tensor)  # cast to fp8, quantize之后就是MXFP8Tensor了
        return new_tensor

