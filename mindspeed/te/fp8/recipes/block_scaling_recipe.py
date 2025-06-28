import math
import torch
import torch_npu

from mindspeed.te.fp8 import Float8Tensor
from mindspeed.te.fp8.recipes.recipe import Recipe, RecipeConfig
from mindspeed.te.fp8.scale_data import ScaleData


class BlockScalingRecipe(Recipe):

    def __init__(self, key, recipe_config: RecipeConfig, tensor_shape) -> None:
        super().__init__(key, recipe_config, tensor_shape)
        if self.config.block_dim is None:
            self.config.block_dim = (128, 128)

        self.scale = ScaleData(recipe_config, self.get_scale_len(tensor_shape))

    def get_scale_len(self, shape):
        if len(shape) != 2:
            # 3d to 2d shape
            shape = [shape[0] * shape[1], shape[2]]
        return [math.ceil(shape[0] / self.config.block_dim[0]),
                math.ceil(shape[1] / self.config.block_dim[1])]

    def quantization(self, tensor, scale_data: ScaleData):
        if tensor is None:
            return tensor

        ori_shape = None
        if len(tensor.shape) != 2:
            ori_shape = tensor.shape
            tensor = tensor.view([-1, tensor.shape[-1]])

        y, scale = torch_npu.npu_dynamic_block_quant(tensor, dst_type=self.scale.fp8_format.dtype,
                                                     row_block_size=self.block_dim[0], col_block_size=self.block_dim[1])
        tensor = Float8Tensor(y, self.scale.fp8_format, 1 / scale, scale, tensor.dtype)

        if ori_shape is not None:
            tensor = tensor.view(ori_shape)
        return tensor

    def dequantization(self, tensor):
        raise RuntimeError('Block scaling has no dequantization method')


    def fp8_block_scaling_matmul(self, inputs, weight, transpose=(False, False)):
        # 这里input有可能是2d或者是3d, 所以使用-1位进行判断
        if inputs.shape[-1] != weight.shape[0]:
            raise AssertionError('shape error.')

        output = torch.matmul(inputs.data, weight.data)

        return output.to(dtype=self.scale.ori_dtype)
