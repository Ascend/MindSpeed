# coding=utf-8
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reversed.
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import warnings

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams


def get_tensor_shapes_in_megatron_bridge(
        *,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: int,
        config,
        tp_group: torch.distributed.ProcessGroup,
        cp_group: torch.distributed.ProcessGroup,
):
    """
    Determine right tensor sizes (based on position of rank with respect to split rank) and
    model size.
    """

    tensor_shapes = []
    # Use decoder_seq_length if provided, otherwise use seq_length
    effective_seq_length = decoder_seq_length if decoder_seq_length is not None else seq_length
    effective_seq_length = effective_seq_length // cp_group.size()

    if config.sequence_parallel:
        effective_seq_length = effective_seq_length // tp_group.size()
    tensor_shapes.append((effective_seq_length * micro_batch_size, 1, config.hidden_size))
    return tensor_shapes


def mtp_checkpointed_forward_impl(
        self,
        forward_func,
        hidden_states: Tensor,
        decoder_input: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        rotary_pos_cos: Tensor,
        rotary_pos_sin: Tensor,
        attention_bias: Tensor,
        inference_params: InferenceParams,
        packed_seq_params: PackedSeqParams,
        sequence_len_offset: Tensor,
):
    """Forward with activation checkpointing.

    Non-tensor args (packed_seq_params, inference_params, attention_bias)
    are captured via closure, following the same pattern as transformer_block.py.
    This fixes the TypeError: save_for_backward can only save variables,
    but argument X is of type PackedSeqParams.
    """

    def custom_forward(
            hidden_states,
            decoder_input,
            attention_mask,
            context,
            context_mask,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
    ):
        return self._proj_and_transformer_layer(
            hidden_states=hidden_states,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            context=context,
            context_mask=context_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

    def checkpoint_handler():
        if self.config.fp8:
            from megatron.core.extensions.transformer_engine import te_checkpoint

            return te_checkpoint(
                custom_forward,
                self.config.distribute_saved_activations,
                tensor_parallel.random.get_cuda_rng_tracker,
                parallel_state.get_tensor_model_parallel_group(),
                hidden_states,
                decoder_input,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                sequence_len_offset,
            )
        else:
            return tensor_parallel.checkpoint(
                custom_forward,
                self.config.distribute_saved_activations,
                hidden_states,
                decoder_input,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                sequence_len_offset,
            )

    if self.config.recompute_method == 'uniform':
        if self.config.recompute_num_layers != 1:
            raise ValueError("recompute_num_layers must be 1 for MTP recompute")
        outputs = checkpoint_handler()
    elif self.config.recompute_method == 'block':
        warnings.warn(
            "recompute_method == 'block' is not supported for MTP yet." " Skipping recompute."
        )
        outputs = self._proj_and_transformer_layer(
            hidden_states=hidden_states,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            context=context,
            context_mask=context_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
    else:
        raise ValueError("Invalid activation recompute method.")

    return outputs
