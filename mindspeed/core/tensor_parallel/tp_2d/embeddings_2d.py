# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import torch
from torch import Tensor

from mindspeed.core.models.common.embeddings.rotary_pos_embedding import (
    _get_pos_emb_on_this_tp_y_cp_rank_in_ulysses_cp,
    _get_pos_emb_on_this_cp_rank_in_ulysses_cp,
    _get_pos_emb_on_this_tp_y_cp_rank_in_megatron_cp
)


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, *args, config=None, **kwargs):
        torch.nn.Module.__init__(self)
        self.config = config

    def forward(self, max_seq_len: int, offset: int = 0, packed_seq: bool = False) -> Tensor:
        """Forward pass of RoPE embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): _description_. Defaults to 0.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        seq = (
            torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            + offset
        )

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor
        freqs = torch.outer(seq, self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
                freqs.shape[0], -1
            )
        # emb [seq_length, .., dim]
        emb = emb[:, None, None, :]

        tp_y_cp_sz = self.config.tp_y * self.config.context_parallel_size
        if tp_y_cp_sz > 1 and not packed_seq:
            # slice rotary_pos_emb along sequence dimension and select the parition of the current CP rank
            if getattr(self.config, 'context_parallel_algo', 'ulysses_cp_algo') == 'ulysses_cp_algo':
                emb = _get_pos_emb_on_this_tp_y_cp_rank_in_ulysses_cp(emb, 0)
            else:
                if getattr(self.config, 'attention_mask_type', 'causal') == 'general':
                    emb = _get_pos_emb_on_this_cp_rank_in_ulysses_cp(emb, 0)
                else:
                    emb = _get_pos_emb_on_this_tp_y_cp_rank_in_megatron_cp(emb, 0)
        return emb
