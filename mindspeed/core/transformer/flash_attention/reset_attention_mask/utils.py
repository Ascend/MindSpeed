# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from functools import wraps
from typing import Dict, Optional

import torch

from mindspeed.utils import set_position_ids


def get_batch_on_this_cp_rank_wrapper(fn):
    @wraps(fn)
    def wrapper(batch):
        batch = fn(batch)
        set_position_ids(batch['position_ids'].transpose(0, 1).contiguous())
        return batch

    return wrapper


def eod_gptdataset_getitem(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        
    """Abstract method implementation

    Args:
        idx (Optioal[int]): The index into the dataset

    Returns:
        Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
    """
    if idx is None:
        # Batch padding sequence so the index does not matter
        text, _ = self._query_document_sample_shuffle_indices(0)
    else:
        text, _ = self._query_document_sample_shuffle_indices(idx)

    text = torch.from_numpy(text).long()
    if self.config.add_extra_token_to_sequence:
        tokens = text[:-1].contiguous()
        labels = text[1:].contiguous()
    else:
        tokens = text
        labels = torch.roll(text, shifts=-1, dims=0)
        labels[-1] = self._pad_token_id

    if (
        not self.masks_and_position_ids_are_cacheable
        or not self.masks_and_position_ids_are_cached
    ):
        attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
            tokens,
            self.config.tokenizer.eod,
            self.config.reset_position_ids,
            self.config.reset_attention_mask,
            self.config.eod_mask_loss,
            self.config.create_attention_mask,
            self.config.tokenizer.vocab_size,
        )
        if self.masks_and_position_ids_are_cacheable:
            self.cached_attention_mask = attention_mask
            self.cached_loss_mask = loss_mask
            self.cached_position_ids = position_ids
            self.masks_and_position_ids_are_cached = True
    else:
        attention_mask = self.cached_attention_mask
        loss_mask = self.cached_loss_mask.clone()
        position_ids = self.cached_position_ids

    # For padded sequences, mask the loss
    loss_mask[labels == self._pad_token_id] = 0.0

    # For padded sequences, ensure the embedding layer can map the token ID
    if self.config.tokenizer.vocab_size is not None:
        tokens[tokens == self.config.tokenizer.vocab_size] = 0
        labels[labels == self.config.tokenizer.vocab_size] = 0

    tokens[tokens == self._pad_token_id] = 0
    labels[labels == self._pad_token_id] = 0

    # Batch padding sequence so we mask the loss
    if idx is None:
        loss_mask = torch.zeros_like(loss_mask)

    if self.config.create_attention_mask:
        return {
            "tokens": tokens,
            "labels": labels,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids
        }
    else:
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids
        }


def _get_ltor_masks_and_position_ids(
        data: torch.Tensor,
        eod_token: int,
        reset_position_ids: bool,
        reset_attention_mask: bool,
        eod_mask_loss: bool,
        create_attention_mask: bool,
        vocab_size: int = None,
):
    """Build masks and position id for left to right model.

    Args:
        data (torch.Tensor): The data tenor that holds the tokens from the dataset

        eod_token (int): ID of the token to that is considered the EOD

        reset_position_ids (bool): Switch to reset the document position ID's

        reset_attention_mask (bool): Switch to reset the attention mask

        eod_mask_loss (bool): Switch to enable the EOD mask loss

        create_attention_mask (bool): Switch to enable the attention masks generation. Can be disabled if attention kernel generates masks by itself.

        vocab_size (int, optional): The vocabulary size. If provided, tokens equal to vocab_size
            will have their loss_mask set to 0 and position_ids set to 0.

    Returns:
        torch.Tensor: Attention mask needed to be used for Attention

        torch.Tensor: The mask used for loss value during training

        torch.Tensor: The position ID's of the token
    """
    seq_length = data.numel()

    if create_attention_mask:
        attention_mask = torch.tril(
            torch.ones((seq_length, seq_length), device=data.device)
        ).unsqueeze(0)
    else:
        attention_mask = None

    # Loss mask.
    loss_mask = torch.ones(seq_length, dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    if reset_attention_mask:
        # Find indices where EOD token is.
        eod_index = position_ids[data == eod_token]
        # Detach indices from positions if going to modify positions.
        if reset_position_ids:
            eod_index = eod_index.clone()

    # Handle vocab_size positions: set to 0 and renumber other positions to be continuous
    if vocab_size is not None:
        # Renumber non-vocab_size positions to be continuous starting from 0
        non_vocab_mask = data != vocab_size
        if non_vocab_mask.any():
            position_ids[non_vocab_mask] = torch.arange(non_vocab_mask.sum(), dtype=torch.long, device=data.device)

    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_attention_mask:
        # Loop through EOD indices:
        for j in range(eod_index.numel()):
            i = eod_index[j]
            # Mask attention loss.
            if reset_attention_mask and attention_mask is not None:
                attention_mask[0, (i + 1):, : (i + 1)] = 0
            # Reset positions.
            if reset_position_ids:
                position_ids[(i + 1):] -= position_ids[i] + 1

    if vocab_size is not None:
        # Set loss_mask and positions where data == vocab_size to 0
        loss_mask[data == vocab_size] = 0.0
        position_ids[data == vocab_size] = 0

    if attention_mask is not None:
        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

    seq_length_tensor = torch.tensor([seq_length])
    if eod_index.numel() > 0 and eod_index[-1] == seq_length_tensor - 1:
        actual_seq_len = eod_index + 1
    else:
        actual_seq_len = torch.cat([eod_index + 1, seq_length_tensor]) if eod_index.numel() > 0 else seq_length_tensor

    return attention_mask, loss_mask, (position_ids, actual_seq_len)


def collate_wrapper(fn):	
    @wraps(fn)	
    def wrapper(samples):	
        actual_seq_len = [elem['position_ids'][1] for elem in samples]
        samples = [{key: val if key != 'position_ids' else val[0] for key, val in elem.items()} for elem in samples]
        batch = fn(samples)	
        seq_len = actual_seq_len[0][-1]	
        actual_seq_len = [elem + i * seq_len for i, elem in enumerate(actual_seq_len)]
        batch['actual_seq_len'] = torch.cat(actual_seq_len)
        return batch	

    return wrapper