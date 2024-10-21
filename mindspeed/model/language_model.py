# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
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
from megatron.training import get_args
from megatron.core import mpu, tensor_parallel
from mindspeed.core.tensor_parallel.comm_group_api import TPXCollectiveComm, TPYCollectiveComm
from mindspeed.core.tensor_parallel.comm_autograd_function import (auto_grad_sync_gather_along_first_dim,
                                                                    auto_grad_sync_gather_along_last_dim,
                                                                    auto_grad_scatter_along_first_dim_then_last_dim)


def parallel_lm_logits(
    input_,
    word_embeddings_weight,
    parallel_output,
    bias=None
):
    args = get_args()
    # Parallel logits.
    if args.async_tensor_model_parallel_allreduce or\
            args.sequence_parallel:
        input_parallel = input_
        model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        async_grad_allreduce = args.async_tensor_model_parallel_allreduce and \
            model_parallel and not args.sequence_parallel
    else:
        input_parallel = tensor_parallel.copy_to_tensor_model_parallel_region(input_)
        async_grad_allreduce = False

    if args.use_nd_matmul:
        input_parallel = tensor_parallel.gather_from_tensor_model_parallel_region(input_parallel)

    if args.tp_2d:
        input_parallel = auto_grad_sync_gather_along_first_dim(input_parallel, TPXCollectiveComm)
        input_parallel = auto_grad_sync_gather_along_last_dim(input_parallel, TPYCollectiveComm)

    # Matrix multiply.
    logits_parallel = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(
        input=input_parallel,
        weight=word_embeddings_weight,
        bias=bias,
        gradient_accumulation_fusion=args.gradient_accumulation_fusion,
        async_grad_allreduce=async_grad_allreduce,
        sequence_parallel=args.sequence_parallel)
    # Gather if needed.

    if parallel_output:
        return logits_parallel

    return tensor_parallel.gather_from_tensor_model_parallel_region(logits_parallel)


def embedding_forward_wrapper(forward):
    @wraps(forward)
    def wrapper(self, *args, **kwargs):
        encoder_input = forward(self, *args, **kwargs)
        if get_args().use_nd_matmul:
            encoder_input = tensor_parallel.scatter_to_tensor_model_parallel_region(encoder_input)
        if get_args().tp_2d:
            encoder_input = auto_grad_scatter_along_first_dim_then_last_dim(
                encoder_input, TPXCollectiveComm, TPYCollectiveComm
            )
        return encoder_input
    return wrapper