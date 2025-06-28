import math

import torch
import torch_npu

from mindspeed.te.fp8 import constants
from mindspeed.te.fp8.constants import Format
from mindspeed.te.fp8.float8_tensor import Float8Tensor
from mindspeed.te.fp8.mxfp8_tensor import MXFP8Tensor


def set_amax(tensor: torch.Tensor, block_dim=None):
    if block_dim is None:
        amax = torch.amax(torch.abs(tensor))
    else:
        amax = torch.empty([math.ceil(tensor.shape[0] / block_dim[0]),
                            math.ceil(tensor.shape[1] / block_dim[1])], device=tensor.device)
        i, j = (0, 0)
        i_start, i_end = i * block_dim[0], (i + 1) * block_dim[0]
        j_start, j_end = j * block_dim[1], (j + 1) * block_dim[1]
        amax[i, j].copy_(torch.amax(torch.abs(tensor[i_start:i_end, j_start:j_end])))

    if not hasattr(tensor, 'fp8_amax'):
        setattr(tensor, 'fp8_amax', amax)
    else:
        tensor.fp8_amax.copy_(amax)


def get_scaling_type(fp8_meta, key):
    from mindspeed.te.fp8.recipes.block_scaling_recipe import BlockScalingRecipe
    recipe = fp8_meta.fp8_config.default[0]
    if hasattr(fp8_meta.fp8_config, key) and getattr(fp8_meta.fp8_config, key) is not None:
        recipe = getattr(fp8_meta.fp8_config, key)[0]
    return recipe


def fp8_matmul(inputs, weight, fp8_meta, key, transpose=(False, False)):
    recipe = get_scaling_type(fp8_meta, key[0])
    from mindspeed.te.fp8.recipes import BlockScalingRecipe, MXFP8ScalingRecipe

    if recipe == MXFP8ScalingRecipe:
        if not isinstance(inputs, MXFP8Tensor):
            inputs = fp8_meta.pre_compute(key[0], inputs)
        if not isinstance(weight, MXFP8Tensor):
            weight = fp8_meta.pre_compute(key[1], weight)

        # quant matmul with transpose
        output = inputs.quant_matmul(weight, transpose)

    else:
        if not isinstance(inputs, Float8Tensor):
            inputs = fp8_meta.pre_compute(key[0], inputs)
        if not isinstance(weight, Float8Tensor):
            weight = fp8_meta.pre_compute(key[1], weight)
        inputs = inputs.t() if transpose[0] else inputs
        weight = weight.t() if transpose[1] else weight
        # quant matmul
        output = inputs.quant_matmul(weight)

    return output


class Cast2FP8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quant_tensor, fp8_format):
        if fp8_format == Format.E4M3.value:
            return quant_tensor.to(torch.float8_e4m3fn)
        elif fp8_format == Format.E5M2.value:
            return quant_tensor.to(torch.float8_e5m2)
        elif fp8_format == Format.HiF8.value:
            return torch_npu.HiFloat8Tensor.to_hifloat8(quant_tensor)
        else:
            raise ValueError("Only e4m3, e5m2 and hif8 of the fp8 format are supported.")

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def cast_to_fp8(quant_tensor, fp8_format):
    return Cast2FP8.apply(quant_tensor, fp8_format)
