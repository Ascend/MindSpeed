# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Packed token shifts for the core 0.12 FB-overlap MTP block."""

import torch
from megatron.core import parallel_state
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper


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
    if self.config.context_parallel_size != 1:
        raise NotImplementedError('FB-overlap packed MTP currently requires context_parallel_size=1.')
    assert labels is not None, 'labels must be provided for MTP loss.'
    if loss_mask is None:
        loss_mask = torch.ones_like(labels)

    hidden_states_main_model = hidden_states
    for layer_number, layer in enumerate(self.layers):
        input_ids = roll_tensor_packed(input_ids, packed_seq_params)
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
        labels = roll_tensor_packed(labels, packed_seq_params)
        loss_mask = roll_tensor_packed(loss_mask, packed_seq_params)
        num_tokens = loss_mask.sum()
        mtp_loss = compute_language_model_loss(labels, mtp_logits) * loss_mask
        # Short segments can have no valid targets left at a deeper MTP layer.
        safe_num_tokens = num_tokens.clamp_min(1)
        if self.training:
            MTPLossLoggingHelper.save_loss_to_tracker(
                mtp_loss.sum() / safe_num_tokens,
                layer_number,
                self.config.mtp_num_layers,
                avg_group=parallel_state.get_tensor_and_context_parallel_group(),
            )
        mtp_loss_scale = self.mtp_loss_scaling_factor / self.config.mtp_num_layers
        if self.config.calculate_per_token_loss:
            hidden_states_main_model = MTPLossAutoScaler.apply(hidden_states_main_model, mtp_loss_scale * mtp_loss)
        else:
            hidden_states_main_model = MTPLossAutoScaler.apply(
                hidden_states_main_model, mtp_loss_scale * mtp_loss / safe_num_tokens
            )
    return hidden_states_main_model
