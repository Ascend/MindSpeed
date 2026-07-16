# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import math
from functools import wraps
from typing import Dict, Optional

import torch

from mindspeed.utils import set_position_ids
from mindspeed.args_utils import get_full_args as get_args
from mindspeed.core.context_parallel import mpu
from mindspeed.core.context_parallel.get_batch_utils import get_actual_seq_len as get_runtime_actual_seq_len


def _resolve_cp_comm_type(args):
    """Resolve CP transport without requiring the optional control-plane PR."""
    try:
        from mindspeed.core.context_parallel.model_parallel_utils import get_resolved_cp_comm_type
    except ImportError:
        pass
    else:
        return get_resolved_cp_comm_type(args)

    cp_comm_type = getattr(args, 'cp_comm_type', None)
    if isinstance(cp_comm_type, (list, tuple)):
        values = list(dict.fromkeys(cp_comm_type))
        cp_comm_type = values[0] if len(values) == 1 else None
    if cp_comm_type == 'allgather':
        return 'all_gather'
    if cp_comm_type is not None:
        return cp_comm_type
    return {
        'megatron_cp_algo': 'p2p',
        'ulysses_cp_algo': 'a2a',
        'kvallgather_cp_algo': 'all_gather',
    }.get(getattr(args, 'context_parallel_algo', None))


def _slice_p2p_eod_batch(batch):
    """Build TENPU RingP2P's two chunks per EOD document.

    This is input-layout adaptation only. Ring communication and attention
    execution remain in TENPU. ``actual_seq_len`` is already padded and
    converted to rank-local endpoints by ``get_batch_utils``.
    """
    actual_seq_len = get_runtime_actual_seq_len()
    if actual_seq_len is None:
        raise AssertionError('P2P EOD batch slicing requires actual_seq_len.')
    if actual_seq_len.dtype not in (torch.int32, torch.int64):
        raise AssertionError('P2P EOD cumulative lengths must use an integral dtype.')

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    if cp_size <= 1:
        return batch

    global_endpoints = (actual_seq_len * cp_size).cpu()
    starts = torch.cat((global_endpoints.new_zeros(1), global_endpoints[:-1]))
    document_lengths = global_endpoints - starts
    chunk_divisor = 2 * cp_size
    if torch.any(torch.remainder(document_lengths, chunk_divisor) != 0):
        raise AssertionError(
            f'P2P EOD document lengths must be divisible by 2 * context_parallel_size={chunk_divisor}.'
        )
    chunk_lengths = torch.div(document_lengths, chunk_divisor, rounding_mode='floor')
    first_starts = starts + cp_rank * chunk_lengths
    first_ends = first_starts + chunk_lengths
    second_starts = global_endpoints - (cp_rank + 1) * chunk_lengths
    second_ends = global_endpoints - cp_rank * chunk_lengths
    index = torch.cat(
        [
            item
            for i in range(actual_seq_len.numel())
            for item in (
                torch.arange(first_starts[i], first_ends[i]),
                torch.arange(second_starts[i], second_ends[i]),
            )
        ]
    ).to(device=actual_seq_len.device)

    for key, value in batch.items():
        if key == 'attention_mask' or value is None:
            continue
        flattened = value.reshape(-1, *value.shape[2:])
        batch[key] = flattened.index_select(0, index).reshape(1, -1, *value.shape[2:])
    return batch


def get_batch_on_this_cp_rank_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args_config = get_args()
        is_p2p_eod = (
            getattr(args_config, 'reset_attention_mask', False)
            and getattr(args_config, 'attention_mask_type', None) == 'causal'
            and getattr(args_config, 'context_parallel_size', 1) > 1
            and _resolve_cp_comm_type(args_config) == 'p2p'
        )
        if is_p2p_eod:
            batch = _slice_p2p_eod_batch(args[0])
        else:
            # all_gather remains owned by Megatron/TENPU's native batch path.
            batch = fn(*args, **kwargs)

        position_ids = batch.get('position_ids')
        if position_ids is not None:
            set_position_ids(position_ids.transpose(0, 1).contiguous())
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

    if not self.masks_and_position_ids_are_cacheable or not self.masks_and_position_ids_are_cached:
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
            "position_ids": position_ids,
        }
    else:
        return {"tokens": tokens, "labels": labels, "loss_mask": loss_mask, "position_ids": position_ids}


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
        attention_mask = torch.tril(torch.ones((seq_length, seq_length), device=data.device)).unsqueeze(0)
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
                attention_mask[0, (i + 1) :, : (i + 1)] = 0
            # Reset positions.
            if reset_position_ids:
                position_ids[(i + 1) :] -= position_ids[i] + 1

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
        args = get_args()
        if hasattr(args, 'fix_sub_seq_length') and 0 < args.fix_sub_seq_length <= args.seq_length:
            per_sample_actual_seq_len = build_fixed_subseq_actual_seq_len(args.seq_length, args.fix_sub_seq_length)
            batch['actual_seq_len'] = torch.cat(
                [per_sample_actual_seq_len + sample_idx * args.seq_length for sample_idx in range(len(samples))]
            )

        else:
            seq_len = actual_seq_len[0][-1]
            actual_seq_len = [elem + i * seq_len for i, elem in enumerate(actual_seq_len)]
            batch['actual_seq_len'] = torch.cat(actual_seq_len)
        return batch

    return wrapper


def build_fixed_subseq_actual_seq_len(seq_length, sub_seq_length):
    times = math.ceil(seq_length / sub_seq_length)
    actual_seq_len_list = [sub_seq_length * i for i in range(1, times + 1)]
    actual_seq_len_list[-1] = seq_length
    actual_seq_len = torch.tensor(actual_seq_len_list, dtype=torch.int64)
    return actual_seq_len
