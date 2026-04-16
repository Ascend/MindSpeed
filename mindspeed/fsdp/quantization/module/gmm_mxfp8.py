# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from functools import partial
from typing import List, Optional, Union

import torch
from torch.distributed.tensor import DTensor

import torch_npu

from einops import rearrange
from mindspeed.fsdp.parallel_engine_config import QuantizeConfig
from mindspeed.fsdp.quantization.core.pre_quant_weight import PreQuantWeight
from mindspeed.ops.npu_moe_token_permute import npu_moe_token_permute
from mindspeed.ops.npu_moe_token_unpermute import npu_moe_token_unpermute
from mindspeed.fsdp.quantization.core.post_quant_weight import PostQuantWeight


@torch._dynamo.allow_in_graph
class gmm_with_hp_or_lp_weight(torch.autograd.Function):

    @classmethod
    def gmm_apply(cls, x, weight, bias, tokens_per_expert, config, grad_enabled, to_shape, ori_weight=None):

        if isinstance(tokens_per_expert, list):
            tokens_per_expert = torch.tensor(tokens_per_expert, device="npu", dtype=torch.int64)
        group_list = torch.cumsum(tokens_per_expert, dim=0)
        return cls.apply(x, weight, bias, group_list, grad_enabled, config, to_shape, ori_weight, 0)

    @classmethod
    def forward(cls, ctx, x, weight, bias, group_list, grad_enabled, config: QuantizeConfig, to_shape, ori_weight=None,
                group_list_type=0):

        def get_quantized_weight(weight, grad_enabled, config):
            ctx.weight_bwd, ctx.weight_scale_bwd = None, None

            if isinstance(weight, PostQuantWeight):
                return weight._weight_fwd, weight._scale_fwd

            if isinstance(weight, PreQuantWeight):
                weight = weight._tensor

            if weight.dtype == torch.float32:
                weight = weight.to(torch.bfloat16)

            if grad_enabled:
                ctx.weight_bwd, ctx.weight_scale_bwd, weight_fwd, weight_scale_fwd = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
                    weight,
                    dst_type=config.get_key_dtype("weight"),
                )

            else:
                weight_fwd, weight_scale_fwd = torch_npu.npu_dynamic_mx_quant(weight, axis=-2,
                                                                              dst_type=config.get_key_dtype("weight"))
            return weight_fwd, weight_scale_fwd

        if isinstance(group_list, torch.Tensor):
            if group_list.device.type == "cpu":
                group_list = group_list.npu()
        else:
            group_list = torch.tensor(group_list, device="npu", dtype=torch.int64)

        # get forward/backward weight
        ori_weight = weight if ori_weight is None else ori_weight
        weight = weight.view(to_shape)
        weight_fwd, weight_scale_fwd = get_quantized_weight(weight, grad_enabled, config)

        # input tensor quantization
        x_mxfp8, x_scale = torch_npu.npu_dynamic_mx_quant(x, axis=-1, dst_type=config.get_key_dtype("inputs"))

        output = torch_npu.npu_grouped_matmul(
            [x_mxfp8],
            [weight_fwd],
            bias=bias,
            scale=[weight_scale_fwd],
            per_token_scale=[x_scale],
            group_list=group_list,
            group_type=0,
            output_dtype=x.dtype,
            group_list_type=group_list_type,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )

        ctx.config = config
        ctx.save_for_backward(x, ori_weight, group_list)
        ctx.group_list_type = group_list_type
        ctx.bias = bias
        ctx.to_shape = to_shape
        return output[0]

    @classmethod
    def backward(cls, ctx, grad_outputs):
        x, weight, group_list = ctx.saved_tensors

        if isinstance(weight, DTensor):
            weight = weight.to_local()

        if isinstance(weight, PostQuantWeight):
            weight_bwd, weight_scale_bwd = weight._weight_bwd, weight._scale_bwd
            weight_bwd = weight_bwd.view(ctx.to_shape)
        else:
            weight_bwd, weight_scale_bwd = ctx.weight_bwd, ctx.weight_scale_bwd

        group_list_type = ctx.group_list_type

        grad_bias = None
        if ctx.bias is not None:
            grad_bias = grad_outputs.reshape(-1, grad_outputs.shape[-1]).sum(dim=0)

        grad_mxfp8, grad_scale = torch_npu.npu_dynamic_mx_quant(grad_outputs, axis=-1,
                                                                dst_type=ctx.config.get_key_dtype("grads"))
        grad_x = torch_npu.npu_grouped_matmul(
            [grad_mxfp8],
            [rearrange(weight_bwd, "n h f -> n f h")],
            scale=[rearrange(weight_scale_bwd, "n h f g -> n f h g")],
            per_token_scale=[grad_scale],
            group_list=group_list,
            group_type=0,
            output_dtype=grad_outputs.dtype,
            group_list_type=group_list_type,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )[0]

        x_mxfp8, x_scale = torch_npu.npu_grouped_dynamic_mx_quant(x, group_list.to(torch.int32), round_mode="rint",
                                                                  dst_type=ctx.config.get_key_dtype("inputs"),
                                                                  blocksize=32)
        grad_mxfp8, grad_scale = torch_npu.npu_grouped_dynamic_mx_quant(
            grad_outputs, group_list.to(torch.int32), round_mode="rint", dst_type=ctx.config.get_key_dtype("grads"),
            blocksize=32
        )

        grad_weights = torch_npu.npu_grouped_matmul(
            [x_mxfp8.t()],
            [grad_mxfp8],
            scale=[grad_scale],
            per_token_scale=[rearrange(x_scale, "n h f -> h n f")],
            group_list=group_list,
            group_type=2,
            output_dtype=x.dtype,
            group_list_type=group_list_type,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )[0]

        return grad_x, grad_weights.view(weight.shape), grad_bias, None, None, None, None, None, None


def mx_quant_group_gemm(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tokens_per_expert: Union[List[int], torch.Tensor] = None,
        grad_enabled: bool = True,
        config: QuantizeConfig = None,
        to_shape: list = None,
        ori_weight: torch.Tensor = None,
) -> torch.Tensor:
    """
    Performs group-wise quantized GEMM (General Matrix Multiplication)
    for MoE (Mixture-of-Experts) models, supporting both high-precision
    and low-precision weight formats.

    Args:
        x:  Input tensor of shape [batch_size * tokens_per_expert, in_features].
            Should be in FP32 or FP16 depending on the quantization setup.
        weight: Quantized weight tensor stored in low-precision format (e.g., MXFP8).
                The function automatically handles dequantization and scaling during computation.
        bias: Optional bias tensor of shape [out_features]. If None, no bias is added.
        tokens_per_expert:  List of integers or tensor specifying the number of tokens
                            assigned to each expert. Can be a list[int] or torch.Tensor.
        grad_enabled: Whether to enable gradient computation (True for training, False for inference).
        config: Quantization configuration object .
        to_shape: Target shape to reshape the weight tensor for GEMM.
        ori_weight: Original weight,Used for backward.

    Returns:
        Output tensor of shape [batch_size * tokens_per_expert, out_features],
    """
    return gmm_with_hp_or_lp_weight.gmm_apply(
        x=x,
        weight=weight,
        bias=bias,
        tokens_per_expert=tokens_per_expert,
        config=config,
        grad_enabled=grad_enabled,
        to_shape=to_shape,
        ori_weight=ori_weight
    )


class MXFP8GMM(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = kwargs.pop("config", None)
        self.num_experts = kwargs.pop("num_experts", None)
        self.hidden_dim = kwargs.pop("hidden_size", None)
        self.intermediate_size = kwargs.pop("moe_intermediate_size", None)
        self.act_fn = kwargs.pop("act_fn", None)

    def forward(self, hidden_states, routing_weights=None, selected_experts=None):
        # permute
        permuted_hidden_states, row_ids_map = npu_moe_token_permute(hidden_states, selected_experts.to(torch.int32))
        tokens_per_expert = torch.histc(selected_experts, bins=self.num_experts, min=0, max=self.num_experts)

        fc1_output = mx_quant_group_gemm(
            x=permuted_hidden_states,
            weight=self.gate_up_proj,
            bias=None,
            tokens_per_expert=tokens_per_expert,
            config=self.config,
            grad_enabled=torch.is_grad_enabled(),
            to_shape=[self.num_experts, self.hidden_dim, -1],
            ori_weight=self.gate_up_proj,
        )

        fc1_activation = torch_npu.npu_swiglu(fc1_output, dim=-1)

        fc2_out = mx_quant_group_gemm(
            x=fc1_activation,
            weight=self.down_proj,
            bias=None,
            tokens_per_expert=tokens_per_expert,
            config=self.config,
            grad_enabled=torch.is_grad_enabled(),
            to_shape=[self.num_experts, -1, self.hidden_dim],
            ori_weight=self.down_proj,
        )
        # unpermute
        output = npu_moe_token_unpermute(fc2_out, row_ids_map, probs=routing_weights)
        return output

    def ep_forward(self, hidden_states, tokens_per_expert):

        gate_up_proj = self.gate_up_proj.to_local()
        down_proj = self.down_proj.to_local()

        fc1_output = mx_quant_group_gemm(
            x=hidden_states,
            weight=gate_up_proj,
            bias=None,
            tokens_per_expert=tokens_per_expert,
            config=self.config,
            grad_enabled=torch.is_grad_enabled(),
            to_shape=[self.num_local_experts, self.hidden_dim, -1],
            ori_weight=self.gate_up_proj,
        )

        fc1_activation = torch_npu.npu_swiglu(fc1_output, dim=-1)

        fc2_out = mx_quant_group_gemm(
            x=fc1_activation,
            weight=down_proj,
            bias=None,
            tokens_per_expert=tokens_per_expert,
            config=self.config,
            grad_enabled=torch.is_grad_enabled(),
            to_shape=[self.num_local_experts, -1, self.hidden_dim],
            ori_weight=self.down_proj,
        )
        return fc2_out

    @classmethod
    def from_float(
            cls,
            mod: torch.nn.Module,
            config: Optional[QuantizeConfig] = None,
            name: Optional[str] = None,
    ):
        if config is None:
            config = QuantizeConfig(recipe_name="mxfp8")

        if config.enable_fsdp_low_precision_all_gather:
            with torch.device("meta"):
                new_mod = cls(
                    config=config,
                    num_experts=mod.num_experts,
                    hidden_size=mod.hidden_dim,
                    moe_intermediate_size=mod.intermediate_size,
                    act_fn=mod.act_fn,
                )

            new_mod.gate_up_proj = mod.gate_up_proj
            new_mod.down_proj = mod.down_proj
            new_mod.gate_up_proj = torch.nn.Parameter(
                PreQuantWeight(
                    new_mod.gate_up_proj,
                    partial(weight_quant, dst_type=config.get_key_dtype("weight"),
                            new_shape=(-1, mod.hidden_dim, mod.intermediate_size * 2)),
                    config,
                    mod.gate_up_proj.dtype,
                    name=name,
                ),
                requires_grad=new_mod.gate_up_proj.requires_grad,
            )

            new_mod.down_proj = torch.nn.Parameter(
                PreQuantWeight(
                    new_mod.down_proj,
                    partial(weight_quant, dst_type=config.get_key_dtype("weight"),
                            new_shape=(-1, mod.intermediate_size, mod.hidden_dim)),
                    config,
                    mod.down_proj.dtype,
                    name=name,
                ),
                requires_grad=new_mod.down_proj.requires_grad,
            )

            new_mod._name = name
            return new_mod

        mod.__class__ = cls
        mod.config = config
        mod._name = name
        return mod


def weight_quant(weight, dst_type, new_shape):
    # To maintain consistency with non-low-precision all-gather,
    # reshape the weight to new_shape before quantization.
    original_shape = weight.shape
    weight = weight.reshape(new_shape)

    weight_bwd, weight_scale_bwd, weight_fwd, weight_scale_fwd = torch_npu.npu_dynamic_mx_quant_with_dual_axis(weight,
                                                                                                               dst_type=dst_type)

    weight_fwd = weight_fwd.reshape(original_shape)
    weight_bwd = weight_bwd.reshape(original_shape)
    return weight_fwd, weight_scale_fwd, weight_bwd, weight_scale_bwd
