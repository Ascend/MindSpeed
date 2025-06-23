# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from collections import OrderedDict
from typing import Dict, Literal, Optional
import torch
from torch import Tensor
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk


# pylint: disable=huawei-too-many-arguments
def model_forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
) -> Tensor:
    """Forward function of the GPT Model This function passes the input tensors
    through the embedding layer, and then the decoeder and finally into the post
    processing layer (optional).

    It either returns the Loss values if labels are given  or the final hidden units

    Args:
        runtime_gather_output (bool): Gather output at runtime. Default None means
            `parallel_output` arg in the constructor will be used.
    """
    # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
    # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.
    if input_ids is not None:
        input_ids = input_ids.to(torch.int64)
    # Decoder embedding.
    if decoder_input is not None:
        pass
    elif self.pre_process:
        decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
    else:
        # intermediate stage of pipeline
        # decoder will get hidden_states from encoder.input_tensor
        decoder_input = None

    # Rotary positional embeddings (embedding is None for PP intermediate devices)
    rotary_pos_emb = None
    rotary_pos_cos = None
    rotary_pos_sin = None
    if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
        if not self.training and self.config.flash_decode:
            # Flash decoding uses precomputed cos and sin for RoPE
            rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cos_sin(
                inference_params.max_sequence_length
            )
        else:
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config, packed_seq_params
            )
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None
                           and packed_seq_params.qkv_format == 'thd',
            )

    # Run decoder.
    hidden_states = self.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        packed_seq_params=packed_seq_params,
        **(extra_block_kwargs or {}),
    )

    if not self.post_process:
        return hidden_states

    # logits and loss
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()
    logits, _ = self.output_layer(
        hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
    )

    if has_config_logger_enabled(self.config):
        payload = OrderedDict(
            {
                'input_ids': input_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'decoder_input': decoder_input,
                'logits': logits,
            }
        )
        log_config_to_disk(self.config, payload, prefix='input_and_logits')

    if labels is None:
        # [s b h] => [b s h]
        return logits.transpose(0, 1).contiguous()

    loss = self.compute_language_model_loss(labels, logits)

    return loss