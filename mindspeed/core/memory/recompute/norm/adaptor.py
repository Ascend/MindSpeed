# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright 2024 Huawei Technologies Co., Ltd.
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
import types

from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

from mindspeed.core.memory.recompute.norm.norm_recompute_forward import norm_recompute_forward_impl
from mindspeed.core.memory.recompute.norm.should_recompute import should_recompute_norm
from mindspeed.model.transformer import NoopTransformerLayer


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
    inference_context=None,
    packed_seq_params=None,
    sequence_len_offset=None,
    *,
    inference_params=None,
):
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
    return norm_recompute_forward_impl(self, 
                                       get_cuda_rng_tracker, 
                                       hidden_states, 
                                       attention_mask, 
                                       context, 
                                       context_mask,
                                       rotary_pos_emb, 
                                       rotary_pos_cos,
                                       rotary_pos_sin,
                                       attention_bias,
                                       inference_context,
                                       packed_seq_params,
                                       sequence_len_offset,
                                       inference_params, 
                                       )


def build_norm_recompute_layer_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        self.layer_number = getattr(self, "layer_number", None)
        for layer in self.layers:
            if isinstance(layer, NoopTransformerLayer):
                continue
            if should_recompute_norm(self.layer_number, self.config):
                layer.forward = types.MethodType(norm_recompute_forward_impl, layer)
    return wrapper
