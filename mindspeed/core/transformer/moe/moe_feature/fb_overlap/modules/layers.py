# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
import warnings
from typing import Any, Callable, List, Optional

import torch
import torch.distributed
from mindspeed.core.transformer.moe.moe_feature import (
    prepare_input_tensors_for_wgrad_compute,
    get_global_memory_buffer,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    get_args
)
from .weight_grad_store import WeightGradStore


def linear_backward_wgrad_detach(ctx, grad_output):
    input_, weight = ctx.saved_tensors
    use_bias = ctx.use_bias
    grad_output_buffer = ctx.grad_output_buffer
    wgrad_deferral_limit = ctx.wgrad_deferral_limit

    wgrad_compute = True
    if grad_output_buffer is not None:
        if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
            grad_output_buffer.append(grad_output)
            wgrad_compute = False

    if wgrad_compute:
        if ctx.sequence_parallel and not WeightGradStore.is_decoupleBlock:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input_.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(
                dim_size, input_.dtype, "mpu"
            )
            handle = torch.distributed._all_gather_base(
                all_gather_buffer, input_, group=get_tensor_model_parallel_group(), async_op=True
            )

            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # gather is scheduled before the input gradient computation
            total_input = all_gather_buffer
        else:
            total_input = input_
    grad_input = grad_output.matmul(weight)

    if ctx.sequence_parallel and wgrad_compute and not WeightGradStore.is_decoupleBlock:
        handle.wait()

    if wgrad_compute and not WeightGradStore.is_decoupleBlock:
        grad_output, total_input = prepare_input_tensors_for_wgrad_compute(
            grad_output, total_input
        )

    if ctx.allreduce_dgrad:
        # Asynchronous all-reduce
        handle = torch.distributed.all_reduce(
            grad_input, group=get_tensor_model_parallel_group(), async_op=True
        )
        # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
        # all-reduce is scheduled before the weight gradient computation

    if ctx.sequence_parallel:
        assert not ctx.allreduce_dgrad
        dim_size = list(input_.size())
        sub_grad_input = torch.empty(
            dim_size, dtype=input_.dtype, device=torch.cuda.current_device(), requires_grad=False
        )
        # reduce_scatter
        handle = torch.distributed._reduce_scatter_base(
            sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
        )
        # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
        # reduce scatter is scheduled before the weight gradient computation


    if WeightGradStore.is_decoupleBlock:
        # TODO: remove clone under MLA setting
        WeightGradStore.put(
            total_input.clone().detach(),
            grad_output.clone().detach(),
            weight,
            ctx.sequence_parallel,
            in_row=not ctx.sequence_parallel
        )
        if hasattr(weight, 'grad_added_to_main_grad') and get_args().overlap_grad_reduce:
            weight.skip_grad_accum = True
        grad_weight = None
    else:
        if ctx.gradient_accumulation_fusion:
            if wgrad_compute:
                if weight.main_grad.dtype == torch.float32:
                    from mindspeed.ops.npu_matmul_add import npu_matmul_add_fp32
                    npu_matmul_add_fp32(total_input, grad_output, weight.main_grad)
                elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                    raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

            if hasattr(weight, 'grad_added_to_main_grad'):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                if getattr(weight, 'zero_out_wgrad', False):
                    grad_weight = torch.zeros(
                        weight.main_grad.shape,
                        dtype=input_.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weight = torch.empty(
                        weight.main_grad.shape,
                        dtype=input_.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)

    grad_bias = grad_output.sum(dim=0) if use_bias else None

    if ctx.sequence_parallel:
        handle.wait()
        # Need to return None's as gradient has to flow for all the input arguments
        # provided during forward
        return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None

    if ctx.allreduce_dgrad:
        handle.wait()

    return grad_input, grad_weight, grad_bias, None, None, None, None, None
