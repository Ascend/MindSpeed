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
"""Packed token shifts for the core 0.12 FB-overlap MTP block."""

import torch
from megatron.core import parallel_state
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper
from mindspeed.core.context_parallel.get_batch_utils import get_packed_mtp_cp_layout


def roll_tensor_packed(tensor, packed_seq_params):
    """Shift each segment of a CP=1 batch, whose boundaries cover the flattened input."""
    endpoints = packed_seq_params.cu_seqlens_q.to(device=tensor.device, dtype=torch.long, non_blocking=True)

    # MindSpeed's collate offsets each sample's endpoints into the flattened batch.
    # Slot zero is discarded below: both [end, ...] and [0, end, ...] work.
    ends = torch.zeros(tensor.numel() + 1, dtype=torch.bool, device=tensor.device)
    ends.index_fill_(0, endpoints, True)
    shifted = torch.roll(tensor.reshape(-1), shifts=-1, dims=0)
    shifted.masked_fill_(ends[1:], 0)
    shifted = shifted.reshape_as(tensor)
    return shifted


def _packed_layer_targets(input_ids, labels, loss_mask, packed_seq_params, depth, cp_group):
    if cp_group is None:
        for _ in range(depth):
            input_ids = roll_tensor_packed(input_ids, packed_seq_params)
            labels = roll_tensor_packed(labels, packed_seq_params)
            loss_mask = roll_tensor_packed(loss_mask, packed_seq_params)
            yield input_ids, labels, loss_mask
        return

    positions, doc_ends = get_packed_mtp_cp_layout(input_ids, packed_seq_params)
    total = input_ids.numel() * torch.distributed.get_world_size(cp_group)

    def gather(tensor):
        out = tensor.new_empty(total)
        torch.distributed.all_gather_into_tensor(out, tensor.reshape(-1).contiguous(), group=cp_group)
        return out

    # The order of all_gather is CP-rank order. Invert the actual batch
    # permutation once so a destination can look up a source on any rank.
    rank_order = gather(positions)
    global_to_gathered = torch.empty_like(rank_order)
    global_to_gathered.scatter_(0, rank_order, torch.arange(total, device=positions.device))
    gathered = [gather(tensor) for tensor in (input_ids, labels, loss_mask)]
    for offset in range(1, depth + 1):
        source = positions + offset
        outside_doc = source >= doc_ends
        # Clamp at the batch end for a safe lookup, then mask every target past its document's last real token.
        indices = global_to_gathered.index_select(0, source.clamp_max(total - 1))
        yield tuple(
            tensor.index_select(0, indices).masked_fill_(outside_doc, 0).view_as(input_ids) for tensor in gathered
        )


def mtp_block_packed_forward(
    self,
    input_ids,
    position_ids,
    hidden_states,
    attention_mask,
    labels=None,
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    attention_bias=None,
    inference_params=None,
    packed_seq_params=None,
    sequence_len_offset=None,
    extra_block_kwargs=None,
    runtime_gather_output=None,
    loss_mask=None,
    embedding=None,
    output_layer=None,
    output_weight=None,
    compute_language_model_loss=None,
):
    """Keep core 0.12's MTP forward/loss contract, replacing its three whole-sequence shifts."""
    assert labels is not None, 'labels must be provided for MTP loss.'
    if loss_mask is None:
        loss_mask = torch.ones_like(labels)

    hidden_states_main_model = hidden_states
    cp_size = self.config.context_parallel_size
    cp_group = parallel_state.get_context_parallel_group() if cp_size > 1 else None
    targets = _packed_layer_targets(input_ids, labels, loss_mask, packed_seq_params, len(self.layers), cp_group)
    for layer_number, layer in enumerate(self.layers):
        input_ids, labels, loss_mask = next(targets)
        # Preserve core 0.12's position_ids and its attention/RoPE convention.
        decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)
        hidden_states = layer(
            decoder_input=decoder_input,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **(extra_block_kwargs or {}),
        )
        mtp_logits, _ = output_layer(hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)
        num_tokens = loss_mask.sum()
        mtp_loss = compute_language_model_loss(labels, mtp_logits) * loss_mask
        if cp_group is not None:
            torch.distributed.all_reduce(num_tokens, group=cp_group)
        # Short segments can have no valid targets left at a deeper MTP layer.
        safe_num_tokens = num_tokens.clamp_min(1)
        if self.training:
            # CP shards can have different valid token counts. Sum their losses before computing the logged mean.
            loss_sum = mtp_loss.detach().sum()
            if cp_group is not None:
                torch.distributed.all_reduce(loss_sum, group=cp_group)
            MTPLossLoggingHelper.save_loss_to_tracker(
                loss_sum / safe_num_tokens,
                layer_number,
                self.config.mtp_num_layers,
                avg_group=parallel_state.get_tensor_and_context_parallel_group(),
            )
        mtp_loss_scale = self.mtp_loss_scaling_factor / self.config.mtp_num_layers
        if self.config.calculate_per_token_loss:
            # Core applies per-token normalization in finalize_model_grads.
            hidden_states_main_model = MTPLossAutoScaler.apply(hidden_states_main_model, mtp_loss_scale * mtp_loss)
        else:
            # num_tokens already covers all CP shards. The cp_size factor offsets Core 0.12's DP x CP gradient average.
            hidden_states_main_model = MTPLossAutoScaler.apply(
                hidden_states_main_model, (mtp_loss_scale * cp_size) * mtp_loss / safe_num_tokens
            )
    return hidden_states_main_model
