# Copyright 2024 Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch_npu
from torch.library import impl
from mindspeed.ops._ffn_registration import FFNOpRegistry
from mindspeed.op_builder.builder import AS_LIBRARY

__all__ = ["npu_ffn"]


op_registry = FFNOpRegistry()


@torch.jit.script
def _grouped_relu_ffn(
    x: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    expert_tokens: torch.Tensor,
    group_list_type: int,
):
    # Keep expert counts on the device and submit the stock operations together.
    hidden = torch.ops.npu.npu_grouped_matmul(
        [x],
        [weight1],
        bias=[bias1] if bias1 is not None else None,
        group_list=expert_tokens,
        group_list_type=group_list_type,
        split_item=3,
        group_type=0,
    )[0]
    hidden = torch.relu_(hidden)
    return torch.ops.npu.npu_grouped_matmul(
        [hidden],
        [weight2],
        bias=[bias2] if bias2 is not None else None,
        group_list=expert_tokens,
        group_list_type=group_list_type,
        split_item=3,
        group_type=0,
    )[0]


def _supports_grouped_relu(
    x, weight1, weight2, tokens, bias1, bias2, activation, inner_precise, output_dtype, quantization
):
    return (
        activation == "relu"
        and inner_precise in (None, 0)
        and output_dtype is None
        and all(value is None for value in quantization)
        and x.dtype in (torch.float16, torch.bfloat16)
        and weight1.dtype == weight2.dtype == x.dtype
        and x.dim() == 2
        and weight1.dim() == weight2.dim() == 3
        and weight1.shape[0] == weight2.shape[0] > 1
        and tokens is not None
        and tokens.device == x.device
        and tokens.dtype == torch.int64
        and tokens.dim() == 1
        and tokens.numel() == weight1.shape[0]
        and tokens.is_contiguous()
        and x.is_contiguous()
        and weight1.is_contiguous()
        and weight2.is_contiguous()
        and all(bias is None or (bias.dim() == 2 and bias.is_contiguous()) for bias in (bias1, bias2))
    )


@impl(AS_LIBRARY, "npu_ffn", "PrivateUse1")
def _npu_ffn(
    x,
    weight1,
    weight2,
    activation,
    *,
    expert_tokens=None,
    expert_tokens_index=None,
    bias1=None,
    bias2=None,
    scale=None,
    offset=None,
    deq_scale1=None,
    deq_scale2=None,
    antiquant_scale1=None,
    antiquant_scale2=None,
    antiquant_offset1=None,
    antiquant_offset2=None,
    inner_precise=None,
    output_dtype=None,
):
    if expert_tokens is not None and expert_tokens_index is not None:
        raise ValueError("expert_tokens and expert_tokens_index are mutually exclusive")
    tokens = expert_tokens if expert_tokens is not None else expert_tokens_index
    quantization = (
        scale,
        offset,
        deq_scale1,
        deq_scale2,
        antiquant_scale1,
        antiquant_scale2,
        antiquant_offset1,
        antiquant_offset2,
    )
    if _supports_grouped_relu(
        x, weight1, weight2, tokens, bias1, bias2, activation, inner_precise, output_dtype, quantization
    ):
        return _grouped_relu_ffn(x, weight1, weight2, bias1, bias2, tokens, int(expert_tokens_index is None))
    if expert_tokens is not None:
        expert_tokens = expert_tokens.tolist()
    if expert_tokens_index is not None:
        expert_tokens_index = expert_tokens_index.tolist()
    return torch_npu.npu_ffn(
        x,
        weight1,
        weight2,
        activation,
        expert_tokens=expert_tokens,
        expert_tokens_index=expert_tokens_index,
        bias1=bias1,
        bias2=bias2,
        scale=scale,
        offset=offset,
        deq_scale1=deq_scale1,
        deq_scale2=deq_scale2,
        antiquant_scale1=antiquant_scale1,
        antiquant_scale2=antiquant_scale2,
        antiquant_offset1=antiquant_offset1,
        antiquant_offset2=antiquant_offset2,
        inner_precise=inner_precise,
        output_dtype=output_dtype,
    )


def npu_ffn(*args, **kwargs):
    if torch.compiler.is_compiling():
        return torch.ops.mindspeed.npu_ffn(*args, **kwargs)
    x = args[0] if args else kwargs["x"]
    if x.device.type == "npu":
        # Eager calls avoid redispatching a Python implementation through C++.
        return _npu_ffn(*args, **kwargs)
    return torch.ops.mindspeed.npu_ffn(*args, **kwargs)
