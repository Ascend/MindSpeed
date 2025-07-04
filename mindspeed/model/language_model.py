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
import torch
from megatron.training import get_args
from megatron.core import mpu, tensor_parallel
from mindspeed.core.tensor_parallel.comm_group_api import TPXCollectiveComm, TPYCollectiveComm
from mindspeed.core.tensor_parallel.comm_autograd_function import (auto_grad_sync_gather_along_first_dim,
                                                                    auto_grad_sync_gather_along_last_dim,
                                                                    auto_grad_scatter_along_first_dim_then_last_dim)
from mindspeed.moe.utils import get_slice_indices_from_disorder_to_order, get_slice_indices_from_order_to_disorder


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


class AmpipeEmbeddingRearrange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, embeddings, ampipe_degree):
        seqlen = embeddings.size(0)
        new_indices = get_slice_indices_from_disorder_to_order(seqlen, ampipe_degree, device=embeddings.device)
        embeddings = torch.index_select(embeddings, dim=0, index=new_indices)
        ctx.ampipe_degree = ampipe_degree
        return embeddings

    @staticmethod
    def backward(ctx, grad_input):
        seqlen = grad_input.size(0)
        new_indices = get_slice_indices_from_order_to_disorder(seqlen, ctx.ampipe_degree, device=grad_input.device)
        grad_input = torch.index_select(grad_input, dim=0, index=new_indices)
        return grad_input, None


def embedding_forward_ampipe(self, input_ids, position_ids, tokentype_ids=None):
    # Embeddings.
    words_embeddings = self.word_embeddings(input_ids)
    if self.add_position_embedding:
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
    else:
        embeddings = words_embeddings

    if tokentype_ids is not None:
        assert self.tokentype_embeddings is not None
        embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
    else:
        assert self.tokentype_embeddings is None

    # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
    embeddings = embeddings.transpose(0, 1).contiguous()

    # If the input flag for fp32 residual connection is set, convert for float.
    if self.fp32_residual_connection:
        embeddings = embeddings.float()

    # Dropout.
    if self.sequence_parallel:
        ampipe_degree = get_args().ampipe_degree
        if ampipe_degree > 1:
            embeddings = AmpipeEmbeddingRearrange.apply(embeddings, ampipe_degree)
        embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
        # `scatter_to_sequence_parallel_region` returns a view, which prevents
        # the original tensor from being garbage collected. Clone to facilitate GC.
        # Has a small runtime cost (~0.5%).
        if self.clone_scatter_output_in_embedding:
            embeddings = embeddings.clone()
        with tensor_parallel.get_cuda_rng_tracker().fork():
            embeddings = self.embedding_dropout(embeddings)
    else:
        embeddings = self.embedding_dropout(embeddings)

    return embeddings


def model_parallel_config_post_init_wrapper(init_func):
    @wraps(init_func)
    def wrapper(self, *args, **kwargs):
        if self.pipeline_model_parallel_size > 1:
            if self.pipeline_dtype is None:
                raise ValueError(
                    "When using pipeline parallelism, pipeline_dtype must be specified"
                )

        if self.autocast_dtype is None:
            self.autocast_dtype = self.params_dtype

        if self.defer_embedding_wgrad_compute and self.pipeline_model_parallel_size == 1:
            raise ValueError(
                "Cannot defer embedding wgrad compute when pipeline model parallel is not used"
            )

        if self.defer_embedding_wgrad_compute and not self.gradient_accumulation_fusion:
            raise ValueError(
                "Cannot defer embedding wgrad compute when gradient accumulation fusion is not used"
            )

    return wrapper
