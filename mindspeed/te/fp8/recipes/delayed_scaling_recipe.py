import torch

from mindspeed.te.fp8 import set_amax
from mindspeed.te.fp8.recipes.recipe import Recipe, RecipeConfig
from mindspeed.te.fp8.scale_data import ScaleData
from mindspeed.te.fp8.constants import Format


class DelayedScalingRecipe(Recipe):
    ALL_SCALING = []
    MAX_STREAM = None

    def __init__(self, key, recipe_config: RecipeConfig, tensor_shape) -> None:
        super().__init__(key, recipe_config, tensor_shape)
        if DelayedScalingRecipe.MAX_STREAM is None:
            DelayedScalingRecipe.MAX_STREAM = torch.cuda.Stream()
        self.block_dim = None
        self.scale = ScaleData(recipe_config)
        self.current_interval = 1
        DelayedScalingRecipe.ALL_SCALING.append(self)
        # MAX_STREAM need to wait ScaleData finished the initialization
        DelayedScalingRecipe.MAX_STREAM.wait_stream(torch.cuda.current_stream())

    def finally_step(self):
        torch.cuda.current_stream().wait_stream(DelayedScalingRecipe.MAX_STREAM)
        self.scale.reduce_amax(self.amax_reduce_group)
        self.scale.update_scale()

    def quantization(self, tensor, scale_data: ScaleData):
        if tensor is None:
            return tensor

        if self.current_interval < self.config.interval:
            self.current_interval += 1
        else:
            self.current_interval = 1
            with torch.cuda.stream(DelayedScalingRecipe.MAX_STREAM):
                amax = torch.amax(torch.abs(tensor))
                scale_data.append_amax(amax)

        # first amax will use current max
        if scale_data.amax_history_current_len == 0:
            torch.cuda.current_stream().wait_stream(DelayedScalingRecipe.MAX_STREAM)

            scale_data.append_amax(amax)
            scale_data.reduce_amax(self.amax_reduce_group)
            scale_data.update_scale()

        if not hasattr(tensor, 'is_fp8') or not tensor.is_fp8:  # if dtype is not fp8
            new_tensor = scale_data.quantization(tensor)  # cast to fp8
            tensor = new_tensor

        return tensor

    def dequantization(self, tensor):
        dtype = self.scale.ori_dtype
        return (tensor.to(dtype=dtype) * self.scale.scale_inv).to(dtype=dtype)
