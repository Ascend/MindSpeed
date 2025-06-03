# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from einops import rearrange
import torch
from mindspore import ops
from megatron.training import get_args
from mindspeed.core.pipeline_parallel.fb_overlap.modules.weight_grad_store import WeightGradStore
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32


def get_gmm_weight_grad(inputs, grad_out, group_list, group_list_data_type, weight_param, weight_tensor):
    if WeightGradStore.is_decoupleBlock:
        WeightGradStore.put(
            [inputs, group_list, group_list_data_type],
            grad_out,
            weight_param,
            sequence_parallel=False,
            in_row=False,
        )
        if hasattr(weight_param, 'grad_added_to_main_grad') and get_args().overlap_grad_reduce:
            # When overlap_grad_reduce is True, need to ensure that backward hooks
            # are all run on the main backprop thread to prevent deadlocks. Setup
            # dummy grad_weight tensor to prevent backward hooks from being run
            # in a background thread.
            shape = list(weight_tensor.shape)
            shape[1], shape[2] = shape[2], shape[1]
            weight_param.skip_grad_accum = True
            
        grad_weights = None
    else:
        if get_args().gemm_gradient_accumulation_fusion:
            npu_groupmatmul_add_fp32(inputs, grad_out, group_list, weight_param.main_grad)
            if hasattr(weight_param, 'grad_added_to_main_grad'):
                shape = list(weight_tensor.shape)
                shape[1], shape[2] = shape[2], shape[1]
                if getattr(weight_tensor, 'zero_out_wgrad', False):
                    grad_weights = torch.zeros(
                        shape,
                        dtype=inputs.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weights = torch.empty(
                        shape,
                        dtype=inputs.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                weight_param.grad_added_to_main_grad = True
            else:
                grad_weights = None
        else:
            grad_weights = ops.function.math_func.gmm([inputs.t()], [grad_out], bias=[], group_list=group_list, group_type=2,
                                                              group_list_type=group_list_data_type)[0]

    return grad_weights


class GroupedMatmulWithWeightGradDetach(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight_tensor, weight_param, group_list, in_row=False):
        mm_out = ops.function.math_func.gmm([inputs], [weight_tensor], bias=[], group_list=group_list, group_type=0, group_list_type=0)[0]
        ctx.save_for_backward(inputs, weight_tensor, group_list)
        ctx.weight_param = weight_param
        ctx.in_row = in_row

        return mm_out

    @staticmethod
    def backward(ctx, *grad_outs):
        grad_out = grad_outs[0]
        inputs, weight_tensor, group_list = ctx.saved_tensors
        weight_param = ctx.weight_param
        weight_tensor = rearrange(weight_tensor, 'n h f -> n f h')
        grad_inputs = \
        ops.function.math_func.gmm([grad_out], [weight_tensor], bias=[], group_list=group_list, group_type=0, group_list_type=0)[0]
        grad_weights = get_gmm_weight_grad(inputs, grad_out, group_list, 0, weight_param,
                                           weight_tensor)

        return grad_inputs, grad_weights, None, None, None
