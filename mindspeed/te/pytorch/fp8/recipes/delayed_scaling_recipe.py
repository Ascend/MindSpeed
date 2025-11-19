import dataclasses
import typing
from typing import Optional
import torch
import torch_npu

from mindspeed.te.pytorch.fp8 import Float8Tensor
from mindspeed.te.pytorch.fp8.recipes.recipe import Recipe, RecipeScaling
from mindspeed.te.pytorch.fp8.scale_data import ScaleData
from mindspeed.te.pytorch.fp8.tensor import is_fp8_tensor


class DelayedScalingRecipe(Recipe):
    ALL_SCALING = []
    MAX_STREAM = None

    def __init__(self, key, recipe_config: 'TEDelayedScaling', tensor_shape) -> None:
        super().__init__(key, recipe_config, tensor_shape)
        if DelayedScalingRecipe.MAX_STREAM is None:
            DelayedScalingRecipe.MAX_STREAM = torch.cuda.Stream()
        self.scale = ScaleData(recipe_config, self.fp8_format)

        DelayedScalingRecipe.ALL_SCALING.append(self)
        # MAX_STREAM need to wait ScaleData finished the initialization
        DelayedScalingRecipe.MAX_STREAM.wait_stream(torch.cuda.current_stream())

    def finally_step(self):
        torch.cuda.current_stream().wait_stream(DelayedScalingRecipe.MAX_STREAM)
        self.scale.delayed_recipe_update_scale()

    def quantization(self, tensor: torch.Tensor, key=None):
        if tensor is None:
            return tensor
        if is_fp8_tensor(tensor):  # if dtype is fp8 return
            return tensor
        self.scale.delayed_recipe_update_amax(tensor, DelayedScalingRecipe.MAX_STREAM)
        scale = self.scale.quantization_scale
        quant_tensor = torch_npu.npu_quantize(tensor, scale, zero_points=None, dtype=self.quant_dtype, axis=-1)
        return Float8Tensor.to_float8(quant_tensor, fp8_format=self.fp8_format,
                                      scale=scale, dtype=tensor.dtype)


@dataclasses.dataclass
class TEDelayedScaling(RecipeScaling):
    recipe = DelayedScalingRecipe
    amax_reduce_group: torch.distributed.ProcessGroup = None
    override_linear_precision: Optional[typing.Tuple] = None
