# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from mindspeed.core.memory.recompute.recompute_common import CheckpointWithoutOutput
from mindspeed.mindspore.core.utils import make_viewless_tensor
from mindspeed.core.memory.recompute.norm.should_recompute import should_recompute_norm


# pylint: disable=too-many-arguments
def norm_recompute_forward_impl(
    self,
    get_cuda_rng_tracker,
    hidden_states,
    attention_mask,
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    inference_params=None,
    packed_seq_params=None,
):
    self.layer_number = getattr(self, "layer_number", None)
    is_recompute_norm = should_recompute_norm(self.layer_number, self.config)
    # Residual connection.
    residual = hidden_states

    if is_recompute_norm:
        # Optional Input Layer norm
        self.norm_ckpt1 = CheckpointWithoutOutput(get_cuda_rng_tracker)
        input_layernorm_output = self.norm_ckpt1.checkpoint(self.input_layernorm, False, hidden_states)
    else:
        input_layernorm_output = self.input_layernorm(hidden_states)

    # Self attention.
    attention_output_with_bias = self.self_attention(
        input_layernorm_output,
        attention_mask=attention_mask,
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb,
        packed_seq_params=packed_seq_params,
    )

    if is_recompute_norm:
        self.norm_ckpt1.discard_output()
        if self.training:
            attention_output_with_bias[0].register_hook(self.norm_ckpt1.recompute)

    with self.bias_dropout_add_exec_handler():
        hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
            attention_output_with_bias, residual, self.hidden_dropout
        )

    # Residual connection.
    residual = hidden_states

    # Optional Layer norm after self-attention
    pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

    # Cross attention.
    attention_output_with_bias = self.cross_attention(
        pre_cross_attn_layernorm_output,
        attention_mask=context_mask,
        key_value_states=context,
        inference_params=inference_params,
    )

    if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
        context = attention_output_with_bias["context"]

    with self.bias_dropout_add_exec_handler():
        hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
            attention_output_with_bias, residual, self.hidden_dropout
        )

    # Residual connection.
    residual = hidden_states

    # Optional Layer norm post the cross-attention.
    if is_recompute_norm:
        self.norm_ckpt2 = CheckpointWithoutOutput(get_cuda_rng_tracker)
        pre_mlp_layernorm_output = self.norm_ckpt2.checkpoint(self.pre_mlp_layernorm, False, hidden_states)
    else:
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

    # MLP.
    mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

    if is_recompute_norm and self.training:
        self.norm_ckpt2.discard_output()
        mlp_output_with_bias[0].register_hook(self.norm_ckpt2.recompute)

    with self.bias_dropout_add_exec_handler():
        hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
            mlp_output_with_bias, residual, self.hidden_dropout
        )

    # Jit compiled function creates 'view' tensor. This tensor
    # potentially gets saved in the MPU checkpoint function context,
    # which rejects view tensors. While making a viewless tensor here
    # won't result in memory savings (like the data loader, or
    # p2p_communication), it serves to document the origin of this
    # 'view' tensor.
    output = make_viewless_tensor(
        inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
    )

    # CUDA graph requires returned values to be Tensors
    if self.config.external_cuda_graph and self.training:
        return output
    return output, context
