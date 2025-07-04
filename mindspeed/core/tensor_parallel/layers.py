# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2024, Bytedance Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import wraps
from webbrowser import get
import os
import warnings
from typing import List, Optional, Callable, Any

import torch
import torch_npu
import torch.distributed
import torch.nn.functional as F

from megatron.core.transformer import TransformerConfig
from .comm_group_api import TPXCollectiveComm
from .comm_group_api import TPYCollectiveComm

try:
    import fused_weight_gradient_mlp_cuda
except Exception:
    warnings.warn("failed to generate the npu_matmul_add_fp32")
from torch.nn.parameter import Parameter
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.parameter import Parameter
import megatron.core.parallel_state as ps
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu
)

from megatron.core import parallel_state, tensor_parallel, mpu
from megatron.training import get_args

from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
    _reduce_scatter_along_first_dim,
    _gather_along_first_dim,
    _ReduceFromModelParallelRegion,
)
from megatron.core.tensor_parallel.layers import (
    LinearWithGradAccumulationAndAsyncCommunication,
    linear_with_grad_accumulation_and_async_allreduce,
    linear_with_frozen_weight,
)
from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_pipeline_first_stage,
    get_data_parallel_world_size,
    get_data_parallel_rank,
)
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.parallel_state import get_tensor_model_parallel_world_size
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.utils import VocabUtility, divide, split_tensor_along_last_dim
from megatron.core.utils import (
    make_tp_sharded_tensor_for_checkpoint,
    prepare_input_tensors_for_wgrad_compute
)
from mindspeed.core.parallel_state import (
    get_tensor_model_parallel_group_for_nd1_dim1,
    get_tensor_model_parallel_group_for_nd1_dim2,
    get_tensor_model_parallel_group_for_nd2_dim1,
    get_tensor_model_parallel_group_for_nd2_dim2,
    get_tensor_model_parallel_world_size_for_nd1_dim1,
    get_tensor_model_parallel_world_size_for_nd1_dim2,
    get_tensor_model_parallel_world_size_for_nd2_dim1,
    get_tensor_model_parallel_world_size_for_nd2_dim2
)
from mindspeed.core.weight_grad_store import WeightGradStore
from mindspeed.moe.async_comm_utils import get_fw_ag_output
from mindspeed.moe.utils import get_slice_indices_from_disorder_to_order
from .ascend_turbo.mc2_linears_seq_parallel import RowSeqParallelLinear


def linear_with_grad_accumulation_and_async_allreduce_zero3(
    input,
    weight,
    bias,
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel: bool,
    grad_output_buffer=None,
    need_gather_param_in_bw=False):

    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel,
        grad_output_buffer,
        need_gather_param_in_bw,
    ]

    if not linear_with_grad_accumulation_and_async_allreduce_zero3.warned:
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
            if sequence_parallel:
                warnings.warn(
                    "When using sequence parallelism it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce_zero3.warned = True

            if async_grad_allreduce:
                warnings.warn(
                    "When using async grad allreduce it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce_zero3.warned = True

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
linear_with_grad_accumulation_and_async_allreduce_zero3.warned = False


def linear_forward_zero3_wrapper(forward_func):
    @wraps(forward_func)
    def linear_forward_zero3(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel,
        grad_output_buffer, 
        need_gather_param_in_bw=False):
        
        ctx.need_gather_param_in_bw = need_gather_param_in_bw

        return forward_func(
            ctx,
            input,
            weight,
            bias,
            gradient_accumulation_fusion,
            async_grad_allreduce,
            sequence_parallel,
            grad_output_buffer)
        
    return linear_forward_zero3


def linear_backward_zero3_wrapper(func):
    @wraps(func)
    def linear_backward_zero3(ctx, grad_output):
        ctx.gradient_accumulation_fusion = (ctx.gradient_accumulation_fusion and not ctx.need_gather_param_in_bw)
        grad_input, grad_weight, grad_bias, _, _, _, _ = func(ctx, grad_output)
        if ctx.need_gather_param_in_bw:
            _, weight = ctx.saved_tensors
            weight.full_grad = grad_weight
            grad_weight = None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None
        
    return linear_backward_zero3


def linear_forward_main_grad_wrapper(forward_func):
    @wraps(forward_func)
    def linear_forward_main_grad(ctx,
                                 inputs,
                                 weight,
                                 bias,
                                 gradient_accumulation_fusion,
                                 allreduce_dgrad,
                                 wgrad_deferral_limit,
                                 sequence_parallel,
                                 grad_output_buffer,):
        output = forward_func(ctx,
                              inputs,
                              weight,
                              bias,
                              gradient_accumulation_fusion,
                              allreduce_dgrad,
                              wgrad_deferral_limit,
                              sequence_parallel,
                              grad_output_buffer,)
        ctx.weight = weight
        return output

    return linear_forward_main_grad


def linear_backward_main_grad_wrapper(backward_func):
    @wraps(backward_func)
    def linear_backward_main_grad(ctx, grad_output):
        class NewCtx:
            pass
        new_ctx = NewCtx()
        inputs, _ = ctx.saved_tensors
        for key in dir(ctx):
            if key == 'saved_tensors':
                setattr(new_ctx, 'saved_tensors', (inputs, ctx.weight))
            elif key.startswith('__') or key == 'saved_variables':
                continue
            else:
                try:
                    getattr(ctx, key)
                except AttributeError:
                    continue
                setattr(new_ctx, key, getattr(ctx, key))
        return backward_func(new_ctx, grad_output)

    return linear_backward_main_grad


def parallel_linear_init_zero3_wrapper(func):
    @wraps(func)
    def parallel_linear_init(self, *args, **kwargs):
        global_args = get_args()
        self.enable_zero3 = global_args.enable_zero3
        func(self, *args, **kwargs)
        if self.enable_zero3:
            dp_size = get_data_parallel_world_size()
            dp_rank = get_data_parallel_rank()
            tmp_tensor = self.weight.chunk(dp_size, dim=0)[dp_rank]
            self.weight = Parameter(
                torch.empty(
                    tmp_tensor.shape, dtype=self.config.params_dtype
                )
            )
            self.weight.data.copy_(tmp_tensor)
        setattr(self.weight, 'enable_zero3', self.enable_zero3)
        
    return parallel_linear_init


def column_parallel_linear_forward_zero3(self, input_, weight=None):
    """Forward of ColumnParallelLinear

    Args:
        input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        weight (optional): weight tensor to use, compulsory when
            skip_weight_param_allocation is True.

    Returns:
        - output
        - bias

    """
    if weight is None:
        if self.weight is None:
            raise RuntimeError(
                "weight was not supplied to ColumnParallelLinear forward pass "
                "and skip_weight_param_allocation is True."
            )
        weight = self.weight
    else:
        # Check the weight passed in is the correct shape
        expected_shape = (self.output_size_per_partition, self.input_size)
        if weight.shape != expected_shape:
            raise RuntimeError(
                f"supplied weight's shape is {tuple(weight.shape)}, "
                f"not {expected_shape} as expected"
            )

    if self.config._cpu_offloading_context is not None:
        if self.config._cpu_offloading_context.inside_context == True:
            assert (
                self.config.cpu_offloading == False
            ), "CPU Offloading cannot be enabled while using non-TE modules"

    bias = self.bias if not self.skip_bias_add else None

    if (
        self.async_tensor_model_parallel_allreduce
        or self.sequence_parallel
        or self.explicit_expert_comm
    ):
        input_parallel = input_
    else:
        input_parallel = copy_to_tensor_model_parallel_region(input_)

    if self.config.defer_embedding_wgrad_compute:
        self.embedding_activation_buffer.append(input_parallel)

    # Matrix multiply.
    if not weight.requires_grad:
        self._forward_impl = linear_with_frozen_weight
    else:
        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

    output_parallel = self._forward_impl(
        input=input_parallel,
        weight=weight,
        bias=bias,
        gradient_accumulation_fusion=self.gradient_accumulation_fusion,
        async_grad_allreduce=False
        if self.explicit_expert_comm
        else self.async_tensor_model_parallel_allreduce,
        sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
        grad_output_buffer=self.grad_output_buffer
        if self.config.defer_embedding_wgrad_compute
        else None,
        need_gather_param_in_bw=self.enable_zero3
    )
    if self.gather_output:
        # All-gather across the partitions.
        assert not self.sequence_parallel
        output = gather_from_tensor_model_parallel_region(output_parallel)
    else:
        output = output_parallel
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


def row_parallel_linear_forward_zero3(self, input_):

    if self.config._cpu_offloading_context is not None:
        if self.config._cpu_offloading_context.inside_context == True:
            assert (
                self.config.cpu_offloading == False
            ), "CPU Offloading cannot be enabled while using non-TE modules"

    # Set up backprop all-reduce.
    if self.input_is_parallel:
        input_parallel = input_
    else:
        assert not self.sequence_parallel
        input_parallel = scatter_to_tensor_model_parallel_region(input_)
    # Matrix multiply.
    if not self.weight.requires_grad:
        self._forward_impl = linear_with_frozen_weight
    else:
        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce
    output_parallel = self._forward_impl(
        input=input_parallel,
        weight=self.weight,
        bias=None,
        gradient_accumulation_fusion=self.gradient_accumulation_fusion,
        async_grad_allreduce=False,
        sequence_parallel=False,
        need_gather_param_in_bw=self.enable_zero3
    )

    # All-reduce across all the partitions.
    if self.explicit_expert_comm:
        assert self.skip_bias_add
        output_ = output_parallel
    elif self.sequence_parallel:
        output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
    else:
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
    if not self.skip_bias_add:
        output = (output_ + self.bias) if self.bias is not None else output_
        output_bias = None
    else:
        output = output_
        output_bias = self.bias
    return output, output_bias


def vocab_parallel_embedding_forward(self, input_):
    if self.tensor_model_parallel_size > 1:
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | \
                     (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input *= ~input_mask
    else:
        masked_input = input_
        # Get the embeddings.

    if self.deterministic_mode:
        output_parallel = self.weight[masked_input]
    else:
        # F.embedding currently has a non-deterministic backward function
        # For higher accumulation accuracy for bf16 on NPU.
        output_parallel = F.embedding(masked_input, self.weight)

    # Mask the output embedding.
    if self.tensor_model_parallel_size > 1:
        output_parallel *= ~input_mask[..., None]
    # Reduce across all the model parallel GPUs.
    if self.reduce_scatter_embeddings:
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        output_parallel = output_parallel.transpose(0, 1).contiguous()
        output = reduce_scatter_to_sequence_parallel_region(output_parallel)
    else:
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
    return output


def row_parallel_nocomm_optimizer_wrapper(forward_func):
    @wraps(forward_func)
    def row_parallel_forward(*args, **kwargs):
        global_args = get_args()
        output = forward_func(*args, **kwargs)
        recompute_num_layers = global_args.recompute_num_layers or 0

        def is_need_avoid_infinite_recompute_loop():
            return isinstance(output, tuple) and ((global_args.swap_attention and recompute_num_layers > 0)
                                                  or global_args.adaptive_memory_optimization)

        if is_need_avoid_infinite_recompute_loop():
            output, bias = output
            if bias is not None:
                # where only recompute mlp, training enters an infinite loop, this * 1 fix this bug
                bias = bias * 1
            return output, bias

        return output
    return row_parallel_forward


class LinearWithGradAccumulationAndAsyncCommunicationPipeExperts(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        pipe_experts,
        ampipe_degree
    ):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel
        ctx.grad_output_buffer = grad_output_buffer
        ctx.wgrad_deferral_limit = wgrad_deferral_limit
        ctx.pipe_experts = pipe_experts

        if sequence_parallel:
            global_args = get_args()
            if global_args.use_ascend_mc2 and not pipe_experts:
                from .ascend_turbo.ascend_turbo_cfg import ascend_turbo_cfg
                group = get_tensor_model_parallel_group()
                rank = get_tensor_model_parallel_rank()
                ascend_turbo_cfg.set_world_size(get_tensor_model_parallel_world_size)
                hcomm_info = None

                if torch.__version__ > "2.0":
                    global_rank = torch.distributed.get_global_rank(group, rank)
                    hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)
                else:
                    hcomm_info = group.get_hccl_comm_name(rank)

                x = input.reshape(input.shape[0] * input.shape[1], input.shape[2])
                world_size = ascend_turbo_cfg.get_world_size()
                output, _ = torch_npu.npu_all_gather_base_mm(
                    x,
                    weight.t(),
                    hcomm_info,
                    world_size,
                    bias=bias,
                    gather_index=0,
                    gather_output=(not ascend_turbo_cfg.all_gather_recomputation)
                )
                output = output.view(
                    output.shape[0] // input.shape[1], input.shape[1], output.shape[1]
                )
            elif pipe_experts:
                total_input = get_fw_ag_output()[0]
                output = torch.matmul(total_input, weight.t())
            else:
                world_size = get_tensor_model_parallel_world_size()
                dim_size = list(input.size())
                dim_size[0] = dim_size[0] * world_size

                all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
                torch.distributed._all_gather_base(
                    all_gather_buffer, input, group=get_tensor_model_parallel_group()
                )
                total_input = all_gather_buffer
                output = torch.matmul(total_input, weight.t())
        else:
            total_input = input
            output = torch.matmul(total_input, weight.t())

        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
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

                if ctx.pipe_experts:
                    all_gather_buffer = torch.empty(dim_size, dtype=input.dtype, device=torch.cuda.current_device())
                else:
                    all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")

                handle = torch.distributed._all_gather_base(
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

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
            )
            # reduce_scatter
            handle = torch.distributed._reduce_scatter_base(
                sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
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
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None
        from mindspeed.moe.pipe_experts import get_async_bw_all_gather_count
        if ctx.pipe_experts and get_async_bw_all_gather_count() != 2:
            grad_output.storage().resize_(0)

        if ctx.sequence_parallel:
            handle.wait()
            # Need to return None's as gradient has to flow for all the input arguments
            # provided during forward
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

        if ctx.async_grad_allreduce:
            handle.wait()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


class LinearWithGradAccumulationAndAsyncCommunication_nano(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        wgrad_deferral_limit,
        sequence_parallel,
        pipe_experts,
        is_nano_row,
        is_nano_column,
    ):
        ctx.weight = weight
        ctx.save_for_backward(input)
        ctx.is_nano_row = is_nano_row
        ctx.is_nano_column = is_nano_column
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.wgrad_deferral_limit = wgrad_deferral_limit
        ctx.sequence_parallel = sequence_parallel
        ctx.pipe_experts = pipe_experts
        global_args = get_args()
        if is_nano_row:
            total_input = input
            if sequence_parallel:
                if pipe_experts:
                    output = torch.matmul(total_input, weight.t())
                elif global_args.use_ascend_mc2:
                    from .ascend_turbo.ascend_turbo_cfg import ascend_turbo_cfg
                    rank = get_tensor_model_parallel_rank()
                    ascend_turbo_cfg.set_world_size(get_tensor_model_parallel_world_size)
                    world_size = ascend_turbo_cfg.get_world_size()
                    group = get_tensor_model_parallel_group()
                    hcomm_info = None
                    if torch.__version__ > "2.0":
                        global_rank = torch.distributed.get_global_rank(group, rank)
                        hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)
                    else:
                        hcomm_info = group.get_hccl_comm_name(rank)

                    x = input.reshape(input.shape[0] * input.shape[1], input.shape[2])
                    output = torch_npu.npu_mm_reduce_scatter_base(
                        x, weight.t(), hcomm_info, world_size, reduce_op="sum", bias=bias
                    )
                    ctx.hcomm_info = hcomm_info
                    ctx.world_size = world_size
                    output = output.view(
                        output.shape[0] // input.shape[1], input.shape[1], output.shape[1]
                    )
                    return output
                else:
                    output = torch.matmul(total_input, weight.t())
                    output = _reduce_scatter_along_first_dim(output)
            else:
                output = torch.matmul(total_input, weight.t())
            if bias is not None:
                output = output + bias
            return output

        if sequence_parallel:
            if pipe_experts:
                total_input = get_fw_ag_output()[0]
                output = torch.matmul(total_input, weight.t())
            elif global_args.use_ascend_mc2:
                from .ascend_turbo.ascend_turbo_cfg import ascend_turbo_cfg
                group = get_tensor_model_parallel_group()
                rank = get_tensor_model_parallel_rank()
                ascend_turbo_cfg.set_world_size(get_tensor_model_parallel_world_size)
                hcomm_info = None
                if torch.__version__ > "2.0":
                    global_rank = torch.distributed.get_global_rank(group, rank)
                    hcomm_info = group._get_backend(torch.device('npu')).get_hccl_comm_name(global_rank)
                else:
                    hcomm_info = group.get_hccl_comm_name(rank)
                x = input.reshape(input.shape[0] * input.shape[1], input.shape[2])
                world_size = ascend_turbo_cfg.get_world_size()
                output, _ = torch_npu.npu_all_gather_base_mm(
                    x,
                    weight.t(),
                    hcomm_info,
                    world_size,
                    bias=bias,
                    gather_index=0,
                    gather_output=(not ascend_turbo_cfg.all_gather_recomputation),
                )
                output = output.view(
                    output.shape[0] // input.shape[1], input.shape[1], output.shape[1]
                )
            else:
                world_size = get_tensor_model_parallel_world_size()
                dim_size = list(input.size())
                dim_size[0] = dim_size[0] * world_size
                all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
                torch.distributed._all_gather_base(
                    all_gather_buffer, input, group=get_tensor_model_parallel_group(),
                )
                total_input = all_gather_buffer
                output = torch.matmul(total_input, weight.t())
        else:
            total_input = input
            output = torch.matmul(total_input, weight.t())
            if bias is not None:
                output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        weight = ctx.weight
        use_bias = ctx.use_bias
        sequence_parallel = ctx.sequence_parallel
        pipe_experts = ctx.pipe_experts
        global_args = get_args()
        grad_output_gathered = grad_output
        grad_input = None
        if ctx.is_nano_row:
            if ctx.sequence_parallel:
                if pipe_experts:
                    grad_input = grad_output.matmul(weight)
                elif global_args.use_ascend_mc2:
                    hcomm_info = ctx.hcomm_info
                    world_size = ctx.world_size
                    grad_output_ = grad_output.reshape(
                        grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
                    )
                    grad_input, grad_output_gathered = torch_npu.npu_all_gather_base_mm(
                        grad_output_, weight, hcomm_info, world_size, bias=None, gather_index=0
                    )

                    grad_input = grad_input.view_as(input)
                else:
                    grad_output_gathered = _gather_along_first_dim(grad_output)
                    grad_input = grad_output_gathered.matmul(weight)
            else:
                grad_input = grad_output.matmul(weight)

            if WeightGradStore.is_decoupleBlock:
                if pipe_experts and ctx.sequence_parallel:
                    WeightGradStore.put(
                        input.clone().detach(),
                        None,
                        weight,
                        sequence_parallel,
                        in_row=True,
                        pipe_experts=True
                    )
                else:
                    WeightGradStore.put(
                        input.clone().detach(),
                        grad_output.clone().detach(),
                        weight,
                        sequence_parallel,
                        in_row=True,
                        pipe_experts=False
                    )
                if hasattr(weight, 'grad_added_to_main_grad'):
                    grad_weight = torch.zeros(
                            weight.main_grad.shape,
                            dtype=input.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                else:
                    grad_weight = None
            else:
                total_input = input
                grad_output = grad_output_gathered.contiguous()
                # Convert the tensonr shapes to 2D for execution compatibility
                if len(grad_output.shape) != 2:
                    grad_output = grad_output.view(
                        grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
                    )
                total_input = total_input.view(
                    total_input.shape[0] * total_input.shape[1], total_input.shape[2]
                )
                if ctx.gradient_accumulation_fusion:
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
                    grad_weight = grad_output.t().matmul(total_input)
            grad_bias = grad_output.sum(dim=0) if use_bias else None

            return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

        if WeightGradStore.is_decoupleBlock:
            WeightGradStore.put(
                input.clone().detach(),
                grad_output.clone().detach(),
                weight,
                ctx.sequence_parallel
            )
            if hasattr(weight, 'grad_added_to_main_grad'):
                grad_weight = torch.zeros(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
            else:
                grad_weight = None
        if not WeightGradStore.is_decoupleBlock:
            if ctx.sequence_parallel:
                world_size = get_tensor_model_parallel_world_size()
                dim_size = list(input.size())
                dim_size[0] = dim_size[0] * world_size

                all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
                handle = torch.distributed._all_gather_base(
                    all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
                )

                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # gather is scheduled before the input gradient computation
                total_input = all_gather_buffer
            else:
                total_input = input
        grad_input = grad_output.matmul(weight)

        if not WeightGradStore.is_decoupleBlock:
            if ctx.sequence_parallel:
                handle.wait()

        # Doing gather + slicing during the NeMo forward pass can make this tensor
        # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
        # clones it if it's not contiguous

            grad_output = grad_output.contiguous()
            # Convert the tensor shape to 2D for execution compatibility
            grad_output = grad_output.view(
                grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
            )
            total_input = total_input.view(
                total_input.shape[0] * total_input.shape[1], total_input.shape[2]
            )

        if ctx.async_grad_allreduce:
            # Asynchronous all_reduce
            handle = torch.distributed.all_reduce(
                grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
            )
            # reduce_scatter
            handle = torch.distributed._reduce_scatter_base(
                sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce_scatter is scheduled before the weight gradient computation
        if not WeightGradStore.is_decoupleBlock:
            if ctx.gradient_accumulation_fusion:
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
                grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

        if ctx.async_grad_allreduce:
            handle.wait()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


class LinearWithGradAccumulationAndAsyncCommunicationAmpipe(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        ampipe_degree,
        is_dense_h_to_3h
    ):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.wgrad_deferral_limit = wgrad_deferral_limit
        ctx.grad_output_buffer = grad_output_buffer
        ctx.ampipe_degree = ampipe_degree
        ctx.is_dense_h_to_3h = is_dense_h_to_3h
        global_args = get_args()
        ampipe_tp_sp_comm_overlap = global_args.ampipe_tp_sp_comm_overlap
        ctx.ampipe_tp_sp_comm_overlap = ampipe_tp_sp_comm_overlap

        if sequence_parallel:
            if global_args.use_ascend_mc2 and ampipe_degree <= 1:
                group = get_tensor_model_parallel_group()
                world_size = get_tensor_model_parallel_world_size()
                rank = torch.distributed.get_rank(group)
                hcomm_info = None
                if torch.__version__ > "2.0":
                    global_rank = torch.distributed.get_global_rank(group, rank)
                    hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)
                else:
                    hcomm_info = group.get_hccl_comm_name(rank)
                x = input.reshape(input.shape[0] * input.shape[1], input.shape[2])
                output, all_gather_grad_output = torch_npu.npu_all_gather_base_mm(
                    x,
                    weight.t(),
                    hcomm_info,
                    world_size,
                    bias=bias,
                    gather_index=0,
                    gather_output=False,
                )
                output = output.view(
                    int(output.shape[0] / input.shape[1]), input.shape[1], output.shape[1]
                )
            elif ampipe_degree > 1 and is_dense_h_to_3h:
                input_list = input.chunk(ampipe_degree, dim=0)
                output_list = []
                for i in range(ampipe_degree):
                    input_chunk = input_list[i]
                    world_size = get_tensor_model_parallel_world_size()
                    dim_size = list(input_chunk.size())
                    dim_size[0] = dim_size[0] * world_size

                    all_gather_buffer = torch.empty(dim_size, dtype=input_chunk.dtype,
                                                    device=torch.cuda.current_device())
                    torch.distributed._all_gather_base(
                        all_gather_buffer, input_chunk, group=get_tensor_model_parallel_group()
                    )
                    output_chunk = torch.matmul(all_gather_buffer, weight.t())
                    output_list.append(output_chunk)

                output = torch.cat(output_list, dim=0)
            elif ampipe_degree > 1 and not is_dense_h_to_3h and ampipe_tp_sp_comm_overlap:
                total_input = get_fw_ag_output().pop(0)
                output = torch.matmul(total_input, weight.t())
                if bias is not None:
                    output = output + bias
                total_input.untyped_storage().resize_(0)

            else:
                world_size = get_tensor_model_parallel_world_size()
                dim_size = list(input.size())
                dim_size[0] = dim_size[0] * world_size

                all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
                torch.distributed._all_gather_base(
                    all_gather_buffer, input, group=get_tensor_model_parallel_group()
                )
                total_input = all_gather_buffer
                output = torch.matmul(total_input, weight.t())
        else:
            total_input = input

            output = torch.matmul(total_input, weight.t())
            if bias is not None:
                output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
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
                if ctx.ampipe_degree > 1 and ctx.is_dense_h_to_3h:
                    new_indices = get_slice_indices_from_disorder_to_order(dim_size[0],
                                                                           ctx.ampipe_degree,
                                                                           device=torch.cuda.current_device())
                    grad_output = torch.index_select(grad_output, dim=0, index=new_indices)

                all_gather_buffer = get_global_memory_buffer().get_tensor(
                    dim_size, input.dtype, "mpu"
                )
                handle = torch.distributed._all_gather_base(
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
            handle = torch.distributed._reduce_scatter_base(
                sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
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
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()
            # Need to return None's as gradient has to flow for all the input arguments
            # provided during forward
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

        if ctx.allreduce_dgrad:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


def linear_with_grad_accumulation_and_async_allreduce_moe(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel: bool,
    pipe_experts=False,
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    wgrad_deferral_limit: Optional[int] = 0,
    allreduce_dgrad: bool = None,
    matmul_id: int = 1,
    is_nano_row: bool = False,
    is_nano_column: bool = False,
    ampipe_degree: int = 1,
    is_dense_h_to_3h: bool = False,
) -> torch.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calcluation of the
    weight gradients.

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.

    Args:

        input (torch.Tensor required): input like torch.nn.functional.linear

        weight (torch.Tensor required): weight like torch.nn.functional.linear

        bias (torch.Tensor optional): bias like torch.nn.functional.linear

        gradient_accumulation_fusion (bool required): Perform the gradient
            accumulation fusion, requires the custom CUDA extension
            fused_weight_gradient_mlp_cuda module. To use
            gradient_accumulation_fusion you must install APEX with
            --cpp_ext and --cuda_ext. For example: "pip install
            --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
            " Note that the extension requires CUDA>=11. Otherwise, you
            must turn off gradient accumulation fusion."

        async_grad_allreduce (bool required): Do the allreduce of input
            gradients asyncronously with the computation of weight
            gradients. If sequence_parallel is True, this must be
            False, as no all reduce is performed.

        sequence_parallel (bool required): Indicates that sequence
            parallelism is used and thus in the forward pass the input is
            all gathered, and the backward pass the input gradients are
            reduce scattered.

        grad_output_buffer (List[torch.Tensor] optional): Buffer used to save
            output gradients when embedding table wgrad compute is deferred.
            Defaults to None.
    """
    if allreduce_dgrad is None:
        warnings.warn(
            "async_grad_allreduce is deprecated and will be removed in a future release. use allreduce_dgrad instead."
        )
        allreduce_dgrad = async_grad_allreduce

    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit
    ]

    if not linear_with_grad_accumulation_and_async_allreduce_moe.warned:
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
            if sequence_parallel:
                warnings.warn(
                    "When using sequence parallelism it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce_moe.warned = True

            if allreduce_dgrad:
                warnings.warn(
                    "When using async grad allreduce it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce_moe.warned = True

    if get_args().use_nanopipe and parallel_state.get_pipeline_model_parallel_world_size() > 1 \
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        if get_args().use_nanopipe and (is_nano_row or is_nano_column):
            args = [
                input,
                weight,
                bias,
                gradient_accumulation_fusion,
                wgrad_deferral_limit,
                async_grad_allreduce,
                sequence_parallel,
                pipe_experts,
                is_nano_row,
                is_nano_column
            ]
            return LinearWithGradAccumulationAndAsyncCommunication_nano.apply(*args)
    if pipe_experts:
        return LinearWithGradAccumulationAndAsyncCommunicationPipeExperts.apply(*args, pipe_experts, ampipe_degree)
    if ampipe_degree > 1:
        return LinearWithGradAccumulationAndAsyncCommunicationAmpipe.apply(*args, ampipe_degree, is_dense_h_to_3h)

    if get_args().use_nd_matmul:
        args.append(pipe_experts)
        args.append(matmul_id)
        return LinearWithGradAccumulationAndAsyncCommunication_Nd.apply(*args)

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)


linear_with_grad_accumulation_and_async_allreduce_moe.warned = False


def parallel_linear_init_wrapper(init_func):
    @wraps(init_func)
    def parallel_linear_init_func(self, *args, pipe_experts: bool = False, in_nano: bool = False,
                                  ampipe_degree: int = 1,
                                  is_dense_h_to_3h: bool = False,
                                  **kwargs):
        output = init_func(self, *args, **kwargs)
        self.pipe_experts = pipe_experts
        self.in_nano = in_nano
        self.ampipe_degree = ampipe_degree
        self.is_dense_h_to_3h = is_dense_h_to_3h
        return output
    return parallel_linear_init_func


def row_parallel_moe(self, input_):
    """Forward of RowParallelLinear

    Args:
        input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

    Returns:
        - output
        - bias
    """

    if self.config._cpu_offloading_context is not None:
        if self.config._cpu_offloading_context.inside_context == True:
            assert (
                    self.config.cpu_offloading == False
            ), "CPU Offloading cannot be enabled while using non-TE modules"

    # Set up backprop all-reduce.
    global_args = get_args()
    if global_args.use_ascend_mc2 and not self.pipe_experts and not self.in_nano:
        output = Mc2RowSeqParallelLinear.apply(
            input_, self.weight, None, get_tensor_model_parallel_group()
        )

        if not self.skip_bias_add:
            output = output + self.bias if self.bias is not None else output
            output_bias = None
        else:
            output_bias = self.bias

        return output, output_bias

    if self.input_is_parallel:
        input_parallel = input_
    else:
        assert not self.sequence_parallel
        input_parallel = scatter_to_tensor_model_parallel_region(input_)
    # Matrix multiply.
    if not self.weight.requires_grad:
        self._forward_impl = linear_with_frozen_weight
    else:
        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

    if self.in_nano and self.sequence_parallel:
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel=True,
            pipe_experts=self.pipe_experts,
            is_nano_row=self.in_nano,
        )
        output_ = output_parallel
    elif self.ampipe_degree > 1:
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel=False,
            ampipe_degree=self.ampipe_degree,
            pipe_experts=self.pipe_experts
        )
        ampipe_tp_sp_comm_overlap = get_args().ampipe_tp_sp_comm_overlap
        if ampipe_tp_sp_comm_overlap or self.pipe_experts:
            output_ = output_parallel
        elif self.sequence_parallel:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
    else:
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel=False,
            pipe_experts=self.pipe_experts,
            is_nano_row=self.in_nano,
        )
        # All-reduce across all the partitions or self.pipe_experts
        if self.explicit_expert_comm or self.pipe_experts:
            assert self.skip_bias_add
            output_ = output_parallel
        elif self.sequence_parallel:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
    if not self.skip_bias_add:
        output = (output_ + self.bias) if self.bias is not None else output_
        output_bias = None
    else:
        output = output_
        output_bias = self.bias
    return output, output_bias


def column_parallel_moe(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None):
    """Forward of ColumnParallelLinear

    Args:
        input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        weight (optional): weight tensor to use, compulsory when
            skip_weight_param_allocation is True.

    Returns:
        - output
        - bias

    """
    if weight is None:
        if self.weight is None:
            raise RuntimeError(
                "weight was not supplied to ColumnParallelLinear forward pass "
                "and skip_weight_param_allocation is True."
            )
        weight = self.weight
    else:
        # Check the weight passed in is the correct shape
        expected_shape = (self.output_size_per_partition, self.input_size)
        if weight.shape != expected_shape:
            raise RuntimeError(
                f"supplied weight's shape is {tuple(weight.shape)}, "
                f"not {expected_shape} as expected"
            )

    if self.config._cpu_offloading_context is not None:
        if self.config._cpu_offloading_context.inside_context == True:
            assert (
                self.config.cpu_offloading == False
            ), "CPU Offloading cannot be enabled while using non-TE modules"

    bias = self.bias if not self.skip_bias_add else None

    if (
        self.allreduce_dgrad
        or self.sequence_parallel
        or self.explicit_expert_comm
    ):
        input_parallel = input_
    else:
        input_parallel = copy_to_tensor_model_parallel_region(input_)

    if self.config.defer_embedding_wgrad_compute:
        if (
                self.config.wgrad_deferral_limit == 0
                or len(self.embedding_activation_buffer) < self.config.wgrad_deferral_limit
        ):
            self.embedding_activation_buffer.append(input_parallel)

    # Matrix multiply.
    if not weight.requires_grad:
        self._forward_impl = linear_with_frozen_weight
    else:
        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

    output_parallel = self._forward_impl(
        input=input_parallel,
        weight=weight,
        bias=bias,
        gradient_accumulation_fusion=self.gradient_accumulation_fusion,
        async_grad_allreduce=False
        if self.explicit_expert_comm
        else self.allreduce_dgrad,
        sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
        grad_output_buffer=(
            self.grad_output_buffer if self.config.defer_embedding_wgrad_compute else None
        ),
        wgrad_deferral_limit=(
            self.config.wgrad_deferral_limit
            if self.config.defer_embedding_wgrad_compute
            else None
        ),
        pipe_experts=self.pipe_experts,
        is_nano_column=self.in_nano,
        ampipe_degree=self.ampipe_degree,
        is_dense_h_to_3h=self.is_dense_h_to_3h
    )
    if self.gather_output:
        # All-gather across the partitions.
        assert not self.sequence_parallel
        output = gather_from_tensor_model_parallel_region(output_parallel)
    else:
        output = output_parallel
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


class Mc2RowSeqParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, group):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None

        from .ascend_turbo.ascend_turbo_cfg import ascend_turbo_cfg
        rank = get_tensor_model_parallel_rank()
        ascend_turbo_cfg.set_world_size(get_tensor_model_parallel_world_size)
        world_size = ascend_turbo_cfg.get_world_size()
        hcomm_info = None

        if torch.__version__ > "2.0":
            global_rank = torch.distributed.get_global_rank(group, rank)
            hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(
                global_rank
            )
        else:
            hcomm_info = group.get_hccl_comm_name(rank)

        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])

        output = torch_npu.npu_mm_reduce_scatter_base(
            x, weight.t(), hcomm_info, world_size, reduce_op="sum", bias=bias
        )

        ctx.hcomm_info = hcomm_info
        ctx.world_size = world_size

        output = output.view(
            output.shape[0] // input_.shape[1], input_.shape[1], output.shape[1]
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        hcomm_info = ctx.hcomm_info
        world_size = ctx.world_size

        grad_output_ = grad_output.reshape(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )

        grad_input, all_gather_grad_output = torch_npu.npu_all_gather_base_mm(
            grad_output_, weight, hcomm_info, world_size, bias=None, gather_index=0
        )
        grad_input = grad_input.view_as(input_)

        total_input = input_
        total_input = total_input.view(
            total_input.shape[0] * total_input.shape[1], total_input.shape[2]
        )
        grad_weight = all_gather_grad_output.t().matmul(total_input)

        is_grad_bias_needed = ctx.needs_input_grad[2]
        if is_grad_bias_needed and ctx.use_bias:
            grad_bias = (
                grad_output.sum(dim=0)
                if grad_output.is_contiguous()
                else grad_output.t().sum(dim=1)

            )
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None


def _initialize_affine_weight_cpu_2d(weight, partition_dim, stride=1, return_master_weight=False, *,
                                     config: TransformerConfig):
    """Initialize affine weight for model parallel when use tp-2d"""
    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )

    if partition_dim == 1:
        row_num = TPYCollectiveComm.get_comm_group_world_size()
        col_num = TPXCollectiveComm.get_comm_group_world_size()
    else:
        row_num = TPXCollectiveComm.get_comm_group_world_size()
        col_num = TPYCollectiveComm.get_comm_group_world_size()

    # Initialize master weight
    split_input_size, split_output_size = weight.size()
    input_size = split_input_size * row_num
    output_size = split_output_size * col_num

    master_weight = torch.empty(input_size, output_size, dtype=torch.float, requires_grad=False)
    config.init_method(master_weight)

    master_weight = master_weight.to(dtype=config.params_dtype)

    x = TPXCollectiveComm.get_comm_rank()
    y = TPYCollectiveComm.get_comm_rank()

    rows = torch.chunk(master_weight, row_num, dim=0)
    if partition_dim == 1:
        row_idx = y
        col_idx = x
    else:
        row_idx = x
        col_idx = y

    row = rows[row_idx]
    cols = torch.chunk(row, col_num, dim=1)
    final_weight = cols[col_idx].contiguous()
    weight.data.copy_(final_weight)

    if return_master_weight:
        return master_weight


def _initialize_affine_weight_cpu_nd(
    weight,
    output_size,
    input_size,
    input_size_per_partition,
    output_size_per_partition,
    init_method,
    stride=1,
    return_master_weight=False,
    *,
    params_dtype=torch.float32
):
    """Initialize affine weight for model parallel when use nd-matmul"""
    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=0, stride=stride
    )

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size, dtype=torch.float, requires_grad=False)
    init_method(master_weight)

    master_weight = master_weight.to(dtype=params_dtype)
    # Split and copy
    rank = ps.get_tensor_model_parallel_rank()
    world_size = ps.get_tensor_model_parallel_world_size()

    def compute_target_rank(rank, row_num, col_num):
        return rank % row_num * col_num + rank // row_num

    # The weight positions of nd and megatron are different. So weight needs to be rearranged.
    # This rearrangement is only to make the calculations of nd and megatron consistent.
    # Even if this rearrangement is removed, it will not affect the correctness of nd calculation.
    row_num = input_size // input_size_per_partition
    col_num = output_size // output_size_per_partition
    weight_list = torch.split(master_weight, master_weight.size()[0] // world_size, dim=0)
    tensor_list = [weight_list[compute_target_rank(i, row_num, col_num)] for i in range(world_size)]
    master_weight = torch.cat(tensor_list, dim=0)

    weight_list_1 = torch.split(master_weight, input_size_per_partition, dim=1)
    weight_1 = weight_list_1[rank // col_num]
    weight_list_2 = torch.split(weight_1, output_size_per_partition, dim=0)
    my_weight_list = weight_list_2[rank % col_num:: world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=0, out=weight)
    if return_master_weight:
        return master_weight
    return None


class LinearWithGradAccumulationAndAsyncCommunication_Nd(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        wgrad_deferral_limit,
        sequence_parallel,
        grad_output_buffer,
        pipe_experts,
        matmul_id,
    ):
        if sequence_parallel:
            raise AssertionError(
                'Nd_matmul cannot be used with sequence_parallel.'
                'If you want to train long sequences, '
                'you can use ulysess or context_parallel that is compatible with nd_matmul.'
            )
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.wgrad_deferral_limit = wgrad_deferral_limit
        ctx.sequence_parallel = sequence_parallel
        ctx.save_for_backward(input, weight)

        if matmul_id == 1:
            world_size1 = get_tensor_model_parallel_world_size_for_nd1_dim1()
            comm_group1 = get_tensor_model_parallel_group_for_nd1_dim1()
            world_size2 = get_tensor_model_parallel_world_size_for_nd1_dim2()
            comm_group2 = get_tensor_model_parallel_group_for_nd1_dim2()
        else:
            world_size1 = get_tensor_model_parallel_world_size_for_nd2_dim1()
            comm_group1 = get_tensor_model_parallel_group_for_nd2_dim1()
            world_size2 = get_tensor_model_parallel_world_size_for_nd2_dim2()
            comm_group2 = get_tensor_model_parallel_group_for_nd2_dim2()

        ctx.world_size1 = world_size1
        ctx.comm_group1 = comm_group1
        ctx.world_size2 = world_size2
        ctx.comm_group2 = comm_group2

        last_dim = input.dim() - 1
        total_input_list = [torch.empty_like(input) for _ in range(world_size1)]
        torch.distributed.all_gather(total_input_list, input, group=comm_group1)
        total_input = torch.cat(total_input_list, dim=last_dim)

        output_parallel = torch.matmul(total_input, weight.t())
        output_parallel = output_parallel.transpose(0, 2)

        dim_size = list(output_parallel.size())
        dim_size[0] //= world_size2
        output = torch.empty(dim_size, dtype=output_parallel.dtype, device=torch.cuda.current_device())
        torch.distributed._reduce_scatter_base(
            output, output_parallel.contiguous(), group=comm_group2
        )
        output = output.transpose(0, 2).contiguous()
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        world_size1 = ctx.world_size1
        comm_group1 = ctx.comm_group1
        world_size2 = ctx.world_size2
        comm_group2 = ctx.comm_group2
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        last_dim = grad_output.dim() - 1

        grad_output_ag_list = [torch.empty_like(grad_output) for _ in range(world_size2)]
        torch.distributed.all_gather(grad_output_ag_list, grad_output.contiguous(), group=comm_group2)
        grad_output_ag = torch.cat(grad_output_ag_list, dim=last_dim)

        total_input_list = [torch.empty_like(input) for _ in range(world_size1)]
        handle1 = torch.distributed.all_gather(total_input_list, input, group=comm_group1, async_op=True)

        grad_bias = grad_output_ag.view(
            grad_output_ag.shape[0] * grad_output_ag.shape[1], grad_output_ag.shape[2]
        ).sum(dim=0) if use_bias else None

        grad_input = grad_output_ag.matmul(weight)

        grad_input = grad_input.transpose(0, 2)
        dim_size = list(grad_input.size())
        dim_size[0] = dim_size[0] // world_size1

        handle1.wait()
        total_input = torch.cat(total_input_list, dim=last_dim)

        grad_input_rs = torch.empty(dim_size, dtype=grad_input.dtype, device=torch.cuda.current_device())

        handle2 = torch.distributed._reduce_scatter_base(
            grad_input_rs, grad_input.contiguous(), group=comm_group1, async_op=True
        )

        grad_output_ag = grad_output_ag.view(
            grad_output_ag.shape[0] * grad_output_ag.shape[1], grad_output_ag.shape[2]
        )
        total_input = total_input.view(
            total_input.shape[0] * total_input.shape[1], total_input.shape[2]
        )

        if ctx.gradient_accumulation_fusion:
            if weight.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    total_input, grad_output_ag, weight.main_grad
                )
            elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                    total_input, grad_output_ag, weight.main_grad
                )
            else:
                raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

            if hasattr(weight, 'grad_added_to_main_grad'):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
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
            grad_weight = grad_output_ag.t().matmul(total_input)

        handle2.wait()
        grad_input_rs = grad_input_rs.transpose(0, 2).contiguous()
        return grad_input_rs, grad_weight, grad_bias, None, None, None, None, None, None, None, None


class Nd_ParallelLinear(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,  # Not used
        matmul_id: int = 1,
    ):
        """Nd_ParallelLinear is used to replace the columnParallelLinear and RowParallelLinear in Megatron TP.

        Args:
            matmul_id: which GEMM operation within the attention or FFN block.
                       if matmul_id is 1 in attention, which represents GEMM for compute QKV.
        """
        super(Nd_ParallelLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        if matmul_id == 1:
            self.world_size_dim1 = get_tensor_model_parallel_world_size_for_nd1_dim1()
            self.world_size_dim2 = get_tensor_model_parallel_world_size_for_nd1_dim2()
        else:
            self.world_size_dim1 = get_tensor_model_parallel_world_size_for_nd2_dim1()
            self.world_size_dim2 = get_tensor_model_parallel_world_size_for_nd2_dim2()

        self.matmul_id = matmul_id
        self.input_size_per_partition = divide(input_size, self.world_size_dim2)
        self.output_size_per_partition = divide(output_size, self.world_size_dim1)

        self.skip_bias_add = skip_bias_add
        self.config = config
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            raise RuntimeError(
                'Nd_matmul cannot be used with sequence_parallel.'
                'If you want to train long sequences, '
                'you can use ulysess or context_parallel that is compatible with nd_matmul.'
            )

        if config.use_cpu_initialization:
            self.weight = torch.nn.Parameter(
                torch.empty(self.output_size, self.input_size_per_partition, dtype=config.params_dtype)
            )

            if config.perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu_nd(
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.input_size_per_partition,
                    self.output_size_per_partition,
                    init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                    params_dtype=config.params_dtype
                )
        else:
            self.weight = torch.nn.Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=1,
                    stride=stride,
                    expert_parallel=(self.is_expert and self.expert_parallel)
                )

        setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))

        if bias:
            if config.use_cpu_initialization:
                self.bias = torch.nn.Parameter(
                    torch.empty(self.output_size, dtype=config.params_dtype)
                )
            else:
                self.bias = torch.nn.Parameter(
                    torch.empty(
                        self.output_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype
                    )
                )

            if config.perform_initialization:
                with torch.no_grad():
                    self.bias.zero_()

            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))
            setattr(self.bias, 'sequence_parallel', self.sequence_parallel)
        else:
            self.register_parameter('bias', None)

        self.explicit_expert_comm = self.is_expert and (
                self.sequence_parallel or self.expert_parallel
        )

    def forward(self, input_):
        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce_moe
        output_ = self._forward_impl(
            input=input_,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel=False,
            matmul_id=self.matmul_id
        )

        if not self.skip_bias_add:
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias

        return output, output_bias
