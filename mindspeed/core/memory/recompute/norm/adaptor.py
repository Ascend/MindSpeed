# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

from mindspeed.core.memory.recompute.norm.norm_recompute_forward import norm_recompute_forward_impl


# pylint: disable=too-many-arguments
def mindspeed_norm_recompute_forward(
    self,
    hidden_states,
    attention_mask=None,
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    attention_bias=None,
    inference_params=None,
    packed_seq_params=None,):
    """
    Perform a forward pass through the transformer layer.

    This method implements the core computation of a transformer layer, including
    self-attention, cross-attention (if applicable), and feed-forward operations.

    Args:
        hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
            b is batch size, and h is hidden size.
        attention_mask (Tensor): Mask tensor for self-attention.
        context (Tensor, optional): Context tensor for cross-attention.
        context_mask (Tensor, optional): Mask tensor for cross-attention.
        rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
        attention_bias (Tensor, optional): Bias tensor for Q * K.T.
        inference_params (object, optional): Parameters for inference-time optimizations.
        packed_seq_params (object, optional): Parameters for packed sequence processing.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing:
            output (Tensor): Transformed hidden states of shape [s, b, h].
            context (Tensor): Updated context tensor if cross-attention is used,
            otherwise None.
    """
    return norm_recompute_forward_impl(self, get_cuda_rng_tracker, hidden_states, attention_mask, context, context_mask,
                                       rotary_pos_emb, inference_params, packed_seq_params)
