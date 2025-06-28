import torch

from mindspeed.te.fp8.recipes.recipe import Recipe, RecipeConfig
from mindspeed.te.fp8.scale_data import ScaleData


class CurrentScalingRecipe(Recipe):

    def __init__(self, key, recipe_config: RecipeConfig, tensor_shape) -> None:
        super().__init__(key, recipe_config, tensor_shape)
        self.block_dim = None
        self.scale = ScaleData(recipe_config)

    def quantization(self, tensor, scale_data: ScaleData):
        if tensor is None:
            return tensor

        fp8_amax = torch.amax(torch.abs(tensor))
        scale_data.amax.copy_(fp8_amax)
        scale_data.reduce_amax(self.amax_reduce_group, scale_data.amax)
        scale_data.update_scale(current_recipe=True)

        if not hasattr(tensor, 'is_fp8') or not tensor.is_fp8:  # if dtype is not fp8
            new_tensor = scale_data.quantization(tensor)  # cast to fp8, quantize之后就是Float8Tensor了
            tensor = new_tensor

        return tensor

    def dequantization(self, tensor):
        dtype = self.input_scale.ori_dtype
        return (tensor.to(dtype=dtype) * self.scale.scale_inv).to(dtype=dtype)
