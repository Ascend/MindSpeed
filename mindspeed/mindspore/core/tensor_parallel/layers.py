# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.


import os
import warnings
from functools import partial
from typing import Any, Callable, List, Optional, Tuple
import torch
from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
)
from megatron.core.utils import is_torch_min_version
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint, prepare_input_tensors_for_wgrad_compute
if is_torch_min_version("2.4.0a0"):
    custom_fwd = partial(torch.amp.custom_fwd, device_type="cuda")
    custom_bwd = partial(torch.amp.custom_bwd, device_type="cuda")
else:
    custom_fwd = torch.cuda.amp.custom_fwd
    custom_bwd = torch.cuda.amp.custom_bwd

if is_torch_min_version("1.13.0"):
    dist_all_gather_func = torch.distributed.all_gather_into_tensor
    dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
else:
    dist_all_gather_func = torch.distributed._all_gather_base
    dist_reduce_scatter_func = torch.distributed._reduce_scatter_base

_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False


@staticmethod
@custom_bwd
def backward(ctx, grad_output):
    """Backward."""
    input, weight = ctx.saved_tensors
    use_bias = ctx.use_bias
    grad_output_buffer = ctx.grad_output_buffer
    wgrad_deferral_limit = ctx.wgrad_deferral_limit

    wgrad_compute = True
    if grad_output_buffer is not None:
        if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
            grad_output_buffer.append(grad_output)
            wgrad_compute = False

    if wgrad_compute:
        if ctx.sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(
                dim_size, input.dtype, "mpu"
            )
            handle = dist_all_gather_func(
                all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
            )

            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # gather is scheduled before the input gradient computation
            total_input = all_gather_buffer
        else:
            total_input = input
    grad_input = grad_output.matmul(weight)

    if ctx.sequence_parallel and wgrad_compute:
        handle.wait()

    if wgrad_compute:
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
        dim_size = list(input.size())
        sub_grad_input = torch.empty(
            dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
        )
        # reduce_scatter
        handle = dist_reduce_scatter_func(
            sub_grad_input.to(grad_input.dtype), grad_input, group=get_tensor_model_parallel_group(), async_op=True
        )
        # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
        # reduce scatter is scheduled before the weight gradient computation

    if ctx.gradient_accumulation_fusion:
        if wgrad_compute:
            if weight.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    total_input, grad_output, weight.main_grad
                )
            elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                    total_input, grad_output, weight.main_grad
                )
            else:
                raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

        if hasattr(weight, 'grad_added_to_main_grad'):
            # When overlap_grad_reduce is True, need to ensure that backward hooks
            # are all run on the main backprop thread to prevent deadlocks. Setup
            # dummy grad_weight tensor to prevent backward hooks from being run
            # in a background thread.
            if getattr(weight, 'zero_out_wgrad', False):
                grad_weight = torch.zeros(
                    weight.main_grad.shape,
                    dtype=input.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
            else:
                grad_weight = torch.empty(
                    weight.main_grad.shape,
                    dtype=input.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
            weight.grad_added_to_main_grad = True
        else:
            grad_weight = None
    else:
        grad_weight = grad_output.t().matmul(total_input.to(grad_output.dtype))
    grad_bias = grad_output.sum(dim=0) if use_bias else None

    if ctx.sequence_parallel:
        handle.wait()
        # Need to return None's as gradient has to flow for all the input arguments
        # provided during forward
        return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None

    if ctx.allreduce_dgrad:
        handle.wait()

    return grad_input, grad_weight, grad_bias, None, None, None, None, None