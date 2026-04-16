# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Defines the prototype UX for converting a model to use mx weights
"""

from typing import Optional, Union
from functools import partial

import torch
import torch_npu

from mindspeed.fsdp.quantization.core.post_quant_weight import PostQuantWeight
from mindspeed.fsdp.quantization.core.pre_quant_weight import PreQuantWeight
from mindspeed.fsdp.parallel_engine_config import QuantizeConfig


@torch._dynamo.allow_in_graph
class matmul_with_hp_or_lp_weight(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            x: torch.Tensor,
            weight: Union[PostQuantWeight | torch.Tensor],
            config: QuantizeConfig,
            grad_enabled: bool,
            bias: torch.Tensor = None,
            name: str = None,
    ):
        orig_shape = x.shape

        x = x.reshape(-1, orig_shape[-1])

        # input tensor quantization
        if grad_enabled:
            x_fwd, x_scale_fwd, x_bwd, x_scale_bwd = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
                x,
                dst_type=config.get_key_dtype("inputs"),
            )
            ctx.x = [x_bwd, x_scale_bwd]
        else:
            x_fwd, x_scale_fwd = torch_npu.npu_dynamic_mx_quant(x, axis=-1, dst_type=config.get_key_dtype("inputs"))
            ctx.x = None

        # weight tensor quantization
        if isinstance(weight, PostQuantWeight):
            weight_fwd, weight_scale_fwd = weight._weight_fwd, weight._scale_fwd
            ctx.weight = weight
            ctx.weight_dtype = weight._orig_dtype
        elif grad_enabled:
            weight_fwd, weight_scale_fwd, weight_bwd, weight_scale_bwd = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
                weight,
                dst_type=config.get_key_dtype("weight"),
            )
            ctx.weight = [weight_bwd, weight_scale_bwd]
            ctx.weight_dtype = weight.dtype
        else:
            weight_fwd, weight_scale_fwd = torch_npu.npu_dynamic_mx_quant(weight, axis=-1,
                                                                          dst_type=config.get_key_dtype("weight"))
            ctx.weight = None
            ctx.weight_dtype = weight.dtype

        ctx.config = config
        ctx.name = name
        ctx.bias = bias

        results = torch_npu.npu_quant_matmul(
            x_fwd,
            weight_fwd.t(),
            weight_scale_fwd.transpose(0, 1),
            pertoken_scale=x_scale_fwd,
            output_dtype=x.dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
            bias=bias,
        )

        # Clear memory
        x_fwd.untyped_storage().resize_(0)
        x_scale_fwd.untyped_storage().resize_(0)
        return results.reshape(*orig_shape[:-1], results.shape[-1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_bwd, x_scale_bwd = ctx.x
        config = ctx.config
        weight_dtype = ctx.weight_dtype

        if isinstance(ctx.weight, PostQuantWeight):
            weight_bwd, weight_scale_bwd = ctx.weight._weight_bwd, ctx.weight._scale_bwd
        else:
            weight_bwd, weight_scale_bwd = ctx.weight

        grad_output_orig_shape = grad_output.shape
        grad_output_reshaped = grad_output.reshape(-1, grad_output_orig_shape[-1])

        # quantization
        grad_di, grad_scale_di, grad_dw, grad_scale_dw = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            grad_output_reshaped,
            dst_type=config.get_key_dtype("grads"),
        )

        grad_bias = None
        if ctx.bias is not None:
            grad_bias = grad_output_reshaped.sum(dim=0)

        grad_x = torch_npu.npu_quant_matmul(
            grad_di,
            weight_bwd,
            weight_scale_bwd,
            pertoken_scale=grad_scale_di,
            output_dtype=grad_output.dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
        )
        grad_x = grad_x.reshape(*grad_output_orig_shape[:-1], grad_x.shape[-1])

        grad_weight = torch_npu.npu_quant_matmul(
            grad_dw.t(),
            x_bwd,
            x_scale_bwd,
            pertoken_scale=grad_scale_dw.transpose(0, 1),
            output_dtype=weight_dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
        )

        # Clear memory
        grad_dw.untyped_storage().resize_(0)
        grad_scale_dw.untyped_storage().resize_(0)
        grad_di.untyped_storage().resize_(0)
        grad_scale_di.untyped_storage().resize_(0)

        x_bwd.untyped_storage().resize_(0)
        x_scale_bwd.untyped_storage().resize_(0)
        return grad_x, grad_weight, None, None, grad_bias, None


def mx_quant_linear(
        x: torch.Tensor,
        weight: torch.Tensor,
        config: QuantizeConfig = None,
        grad_enabled: bool = True,
        bias: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
) -> torch.Tensor:
    """
    Performs forward and backward passes for a quantized linear layer,
    supporting both high-precision and low-precision weight formats.

    Args:
        x:  Input tensor of shape  [batch_size, ..., input_dim].
            Should be in FP32 or BF16 depending on the quantization setup.
        weight: Quantized weight tensor stored in low-precision format (e.g., MXFP8).
                The function automatically handles dequantization and scaling during computation.
        config: Quantization configuration object .
        grad_enabled: Whether to enable gradient computation (True for training, False for inference).
        bias: Optional bias tensor of shape [out_features]. If None, no bias is added.
        name: A descriptive name for the layer, useful for debugging, logging, and visualization.

    Returns:
        Output tensor of shape [batch_size, ..., output_dim], computed using quantized matrix multiplication.
    """
    return matmul_with_hp_or_lp_weight.apply(x, weight, config, grad_enabled, bias, name)


class MXLinear(torch.nn.Linear):
    config: QuantizeConfig

    def __init__(self, *args, **kwargs):
        config = kwargs.pop("config", None)
        super().__init__(*args, **kwargs)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_dtype())

        output = mx_quant_linear(
            x,
            self.weight,
            self.config,
            torch.is_grad_enabled(),
            None,
            self._name,
        )

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    def extra_repr(self):
        if self.config is None:
            return super().extra_repr()
        return f"{super().extra_repr()}, {str(self.config)}"

    @classmethod
    def from_float(
            cls,
            mod: torch.nn.Linear,
            config: Optional[QuantizeConfig] = None,
            name: Optional[str] = None,
    ):
        if config is None:
            config = QuantizeConfig(recipe_name="mxfp8")

        if config.enable_fsdp_low_precision_all_gather:
            with torch.device("meta"):
                new_mod = cls(
                    mod.in_features,
                    mod.out_features,
                    bias=False,
                    config=config,
                )
            new_mod.weight = mod.weight
            new_mod.bias = mod.bias

            new_mod.weight = torch.nn.Parameter(
                PreQuantWeight(
                    new_mod.weight,
                    partial(torch_npu.npu_dynamic_mx_quant_with_dual_axis, dst_type=config.get_key_dtype("weight")),
                    config,
                    mod.weight.dtype,
                    name=name,
                ),
                requires_grad=new_mod.weight.requires_grad,
            )
            new_mod._name = name
            return new_mod

        mod.__class__ = cls
        mod.config = config
        mod._name = name
        return mod
