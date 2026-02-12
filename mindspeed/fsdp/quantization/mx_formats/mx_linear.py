# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Defines the prototype UX for converting a model to use mx weights
"""

from typing import Any, Optional
import torch
import torch_npu

from mindspeed.fsdp.quantization.mx_formats.mx_tensor import MXTensor
from mindspeed.fsdp.quantization.mxfp8_config import MXFP8LinearConfig
from mindspeed.fsdp.quantization.utils import register_quantize_module_handler


@torch._dynamo.allow_in_graph
class mx_mm(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            input_hp: torch.Tensor,
            weight_hp: torch.Tensor,
            mx_tensor: MXTensor,
    ):
        # 低精量化
        ctx.mx_tensor = mx_tensor
        # input @ weight_t = output
        input_orig_shape = input_hp.shape
        input_hp = input_hp.view(-1, input_hp.size(-1))

        input_mx = mx_tensor.to_mxfp8(input_hp, 'inputs')
        weight_mx = mx_tensor.to_mxfp8(weight_hp, 'weight')
        ctx.input_mx = input_mx
        ctx.weight_mx = weight_mx

        input_mxfp8, input_scale = input_mx.data, input_mx.scale
        weight_mxfp8_t, weight_scale_t = weight_mx.data_t, weight_mx.scale_t

        output = torch_npu.npu_quant_matmul(input_mxfp8, weight_mxfp8_t, weight_scale_t,
                                            pertoken_scale=input_scale,
                                            output_dtype=input_hp.dtype, scale_dtype=torch_npu.float8_e8m0fnu,
                                            pertoken_scale_dtype=torch_npu.float8_e8m0fnu, group_sizes=[1, 1, 32])
        if len(input_orig_shape) != 2:
            output = output.reshape(*input_orig_shape[:-1], output.shape[-1])
        if weight_hp.requires_grad:
            output.requires_grad = True
        return output

    @staticmethod
    def backward(ctx, grad_output_hp: torch.Tensor):
        # 低精量化
        input_mx = ctx.input_mx
        weight_mx = ctx.weight_mx
        mx_tensor = ctx.mx_tensor
        ori_dtype = input_mx.ori_dtype
        grad_orig_shape = grad_output_hp.shape
        grad_output_hp = grad_output_hp.view(-1, grad_output_hp.size(-1))

        grads_mx = mx_tensor.to_mxfp8(grad_output_hp, 'grads')
        grads_mxfp8, grads_scale = grads_mx.data, grads_mx.scale
        weight_mxfp8, weight_scale = weight_mx.data, weight_mx.scale
        dx = torch_npu.npu_quant_matmul(grads_mxfp8, weight_mxfp8, weight_scale,
                                        pertoken_scale=grads_scale,
                                        output_dtype=ori_dtype, scale_dtype=torch_npu.float8_e8m0fnu,
                                        pertoken_scale_dtype=torch_npu.float8_e8m0fnu, group_sizes=[1, 1, 32])

        if len(grad_orig_shape) != 2:
            dx = dx.reshape(*grad_orig_shape[:-1], dx.shape[-1])

        grads_mxfp8, grads_scale = grads_mx.data_t, grads_mx.scale_t
        x_mxfp8, x_scale = input_mx.data_t, input_mx.scale_t
        dw = torch_npu.npu_quant_matmul(grads_mxfp8, x_mxfp8, x_scale, pertoken_scale=grads_scale,
                                        output_dtype=ori_dtype, scale_dtype=torch_npu.float8_e8m0fnu,
                                        pertoken_scale_dtype=torch_npu.float8_e8m0fnu, group_sizes=[1, 1, 32])

        return dx, dw, None, None


class MXLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self._mx_tensor = None
        self.config = None

    @classmethod
    @torch.no_grad()
    def from_float(
            cls,
            mod,
            config: Optional[MXFP8LinearConfig] = None,
            *extra_args
    ):
        if config is None:
            config = MXFP8LinearConfig()

        mod.__class__ = MXLinear
        mod.config = config
        mod._mx_tensor = None
        return mod

    @property
    def mx_tensor(self):
        if self._mx_tensor is None:
            self._mx_tensor = MXTensor(self.config)
        return self._mx_tensor

    def forward(self, x):
        w = self.weight
        y = mx_mm.apply(
            x,
            w,
            self.mx_tensor,
        )
        if self.bias is not None:
            y = y + self.bias

        return y


@register_quantize_module_handler(MXFP8LinearConfig)
def _mx_linear_transform(module: torch.nn.Module, config: MXFP8LinearConfig):
    return MXLinear.from_float(module, config=config)
