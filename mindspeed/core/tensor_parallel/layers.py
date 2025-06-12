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


def linear_forward_main_grad_wrapper(forward_func):
    @wraps(forward_func)
    def linear_forward_main_grad(ctx,
                                 inputs,
                                 weight,
                                 bias,
                                 gradient_accumulation_fusion,
                                 allreduce_dgrad,
                                 sequence_parallel,
                                 grad_output_buffer,
                                 wgrad_deferral_limit,
                                 ):
        output = forward_func(ctx,
                              inputs,
                              weight,
                              bias,
                              gradient_accumulation_fusion,
                              allreduce_dgrad,
                              sequence_parallel,
                              grad_output_buffer,
                              wgrad_deferral_limit,
                              )
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


def _initialize_affine_weight_cpu_2d(
    weight,
    output_size,
    input_size,
    input_size_per_partition,
    output_size_per_partition,
    partition_dim,
    init_method,
    stride=1,
    return_master_weight=False,
    *,
    params_dtype=torch.float32
):
    """Initialize affine weight for model parallel when use tp-2d"""
    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
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
    if partition_dim == 0:
        row_num = input_size // input_size_per_partition
        col_num = output_size // output_size_per_partition
    else:
        col_num = input_size // input_size_per_partition
        row_num = output_size // output_size_per_partition
    weight_list = torch.split(master_weight, master_weight.size()[partition_dim] // world_size, dim=partition_dim)
    tensor_list = [weight_list[compute_target_rank(i, row_num, col_num)] for i in range(world_size)]
    master_weight = torch.cat(tensor_list, dim=partition_dim)
    weight_list_1 = torch.split(master_weight, input_size_per_partition, dim=1)
    if partition_dim == 0:
        weight_1 = weight_list_1[rank // col_num]
    else:
        weight_1 = weight_list_1[rank % col_num]
    weight_list_2 = torch.split(weight_1, output_size_per_partition, dim=0)
    if partition_dim == 0:
        my_weight_list = weight_list_2[rank % col_num:: world_size]
    else:
        my_weight_list = weight_list_2[rank // col_num:: world_size]
    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight

