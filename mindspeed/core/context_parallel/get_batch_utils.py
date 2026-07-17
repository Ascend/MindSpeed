# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from mindspeed.core.context_parallel import get_args, mpu
from mindspeed.core.context_parallel.utils import pad_data


_ACTUAL_SEQ_LEN = None


def get_actual_seq_len():
    return _ACTUAL_SEQ_LEN


def set_actual_seq_len(actual_seq_len):
    global _ACTUAL_SEQ_LEN
    _ACTUAL_SEQ_LEN = actual_seq_len


def _broadcast(item):
    if item is not None:
        torch.distributed.broadcast(
            item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group()
        )


def broadcast_dynamic(item, broadcast_src_rank=None, broadcast_group=None):
    def broadcast(tensor):
        if broadcast_src_rank is None:
            _broadcast(tensor)
        else:
            torch.distributed.broadcast(tensor, broadcast_src_rank, group=broadcast_group)

    if item is not None:
        item = item.cuda(non_blocking=True).to(dtype=torch.int64)
        item_len = torch.tensor(item.numel(), device=torch.cuda.current_device())
        broadcast(item_len)
        broadcast(item)
    else:
        item_len = torch.empty((), dtype=torch.int64, device=torch.cuda.current_device())
        broadcast(item_len)
        item = torch.empty([item_len.item()], dtype=torch.int64, device=torch.cuda.current_device())
        broadcast(item)
    return item


def _validate_actual_seq_len(actual_seq_len):
    """Validate the integral EOD cumulative endpoints received from the dataset."""
    if actual_seq_len.dtype not in (torch.int32, torch.int64):
        raise AssertionError(f'EOD cu_seqlens must use an integral dtype, got {actual_seq_len.dtype}.')
    return actual_seq_len


def _uses_p2p_eod_layout(args):
    """Whether TENPU RingP2P needs rank-local TND EOD endpoints."""
    return (
        getattr(args, 'reset_attention_mask', False)
        and getattr(args, 'attention_mask_type', None) == 'causal'
        and getattr(args, 'context_parallel_size', 1) > 1
        and getattr(args, 'context_parallel_algo', None) == 'megatron_cp_algo'
    )


def _prepare_p2p_eod_actual_seq_len(actual_seq_len, batch, args):
    """Pad each EOD document and return exact rank-local TND endpoints.

    TENPU's RingP2P EOD path consumes ``cu_seqlens_q`` directly.  Its final
    endpoint must therefore equal this rank's TND token count, unlike the
    all-gather path which owns its own global-to-local conversion.
    """
    actual_seq_len = _validate_actual_seq_len(actual_seq_len)
    cp_size = args.context_parallel_size
    padded_actual_seq_len = pad_data(actual_seq_len, batch, cp_size, args.tensor_model_parallel_size)
    if torch.any(torch.remainder(padded_actual_seq_len, cp_size) != 0):
        raise AssertionError(f'P2P EOD padded cumulative lengths must be divisible by context_parallel_size={cp_size}.')
    return torch.div(padded_actual_seq_len, cp_size, rounding_mode='floor')


def get_batch_on_this_tp_rank(
    batch: dict[str, torch.Tensor],
    is_sft: bool,
    is_hybrid_cp: bool,
    create_attention_mask_in_dataloader: bool,
    broadcast_src_rank: int,
    broadcast_group: torch.distributed.ProcessGroup,
    cp_size: int,
    tp_rank: int,
    micro_batch_size: int,
    seq_length: int,
    mtp_on_this_rank: bool,
    pipeline_model_parallel_size: int = 1,
    is_pipeline_first_stage: bool = False,
    is_pipeline_last_stage: bool = False,
):
    """Broadcast an already materialized 0.18 batch and retain MindSpeed EOD metadata."""
    args = get_args()

    def broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, broadcast_src_rank, group=broadcast_group)

    def broadcast_cu_seqlens(cu_seqlens):
        n = 0 if cu_seqlens is None else int(cu_seqlens.numel())
        n_tensor = torch.tensor(n, dtype=torch.int64, device=torch.cuda.current_device())
        broadcast(n_tensor)
        if n > 0:
            if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.dtype != torch.int32:
                raise AssertionError('cu_seqlens must be an int32 tensor.')
            broadcast(cu_seqlens)

    if tp_rank == 0:
        if is_hybrid_cp:
            hybrid_cp_seq_length = torch.tensor(
                batch['tokens'].shape[1], dtype=torch.int32, device=torch.cuda.current_device()
            )
            broadcast(hybrid_cp_seq_length)

        if pipeline_model_parallel_size == 1 or mtp_on_this_rank:
            broadcast(batch['tokens'])
            broadcast(batch['labels'])
            broadcast(batch['loss_mask'])
            broadcast(batch['position_ids'])
            if is_sft or is_hybrid_cp:
                broadcast_cu_seqlens(batch['cu_seqlens'])
                broadcast(batch['max_seqlen'])
                if cp_size > 1:
                    broadcast_cu_seqlens(batch['cu_seqlens_padded'])
            if create_attention_mask_in_dataloader:
                broadcast(batch['attention_mask'])
            if is_hybrid_cp:
                broadcast(batch['local_cp_size'])
        elif is_pipeline_first_stage:
            batch['labels'] = None
            batch['loss_mask'] = None
            broadcast(batch['tokens'])
            broadcast(batch['position_ids'])
            if is_sft:
                broadcast_cu_seqlens(batch['cu_seqlens'])
                broadcast(batch['max_seqlen'])
                if cp_size > 1:
                    broadcast_cu_seqlens(batch['cu_seqlens_padded'])
            if create_attention_mask_in_dataloader:
                broadcast(batch['attention_mask'])
        elif is_pipeline_last_stage:
            batch['tokens'] = None
            batch['position_ids'] = None
            broadcast(batch['labels'])
            broadcast(batch['loss_mask'])
            if is_sft:
                broadcast_cu_seqlens(batch['cu_seqlens'])
                broadcast(batch['max_seqlen'])
                if cp_size > 1:
                    broadcast_cu_seqlens(batch['cu_seqlens_padded'])
            if create_attention_mask_in_dataloader:
                broadcast(batch['attention_mask'])
        elif is_sft:
            batch['tokens'] = None
            batch['labels'] = None
            batch['loss_mask'] = None
            batch['position_ids'] = None
            batch['attention_mask'] = None
            broadcast_cu_seqlens(batch['cu_seqlens'])
            broadcast(batch['max_seqlen'])
            if cp_size > 1:
                broadcast_cu_seqlens(batch['cu_seqlens_padded'])
    else:
        if is_hybrid_cp:
            hybrid_cp_seq_length = torch.tensor(0, dtype=torch.int32, device=torch.cuda.current_device())
            broadcast(hybrid_cp_seq_length)
            shape = (micro_batch_size, hybrid_cp_seq_length.item())
        else:
            shape = (micro_batch_size, seq_length)

        tokens = torch.empty(shape, dtype=torch.int64, device=torch.cuda.current_device())
        labels = torch.empty(shape, dtype=torch.int64, device=torch.cuda.current_device())
        loss_mask = torch.empty(shape, dtype=torch.float32, device=torch.cuda.current_device())
        position_ids = torch.empty(shape, dtype=torch.int64, device=torch.cuda.current_device())
        cu_seqlens = None
        cu_seqlens_padded = None
        max_seqlen = None
        attention_mask = None
        local_cp_size = None
        if is_sft or is_hybrid_cp:
            max_seqlen = torch.empty(1, dtype=torch.int32, device=torch.cuda.current_device())
        if create_attention_mask_in_dataloader:
            attention_mask = torch.empty(
                (micro_batch_size, 1, seq_length, seq_length),
                dtype=torch.bool,
                device=torch.cuda.current_device(),
            )
        if is_hybrid_cp:
            local_cp_size = torch.empty(1, dtype=torch.int32, device=torch.cuda.current_device())

        def receive_cu_seqlens():
            n = torch.empty((), dtype=torch.int64, device=torch.cuda.current_device())
            broadcast(n)
            n = int(n.item())
            if n == 0:
                return None
            value = torch.empty((1, n), dtype=torch.int32, device=torch.cuda.current_device())
            broadcast(value)
            return value

        if pipeline_model_parallel_size == 1 or mtp_on_this_rank:
            broadcast(tokens)
            broadcast(labels)
            broadcast(loss_mask)
            broadcast(position_ids)
            if is_sft or is_hybrid_cp:
                cu_seqlens = receive_cu_seqlens()
                broadcast(max_seqlen)
                if cp_size > 1:
                    cu_seqlens_padded = receive_cu_seqlens()
            if create_attention_mask_in_dataloader:
                broadcast(attention_mask)
            if is_hybrid_cp:
                broadcast(local_cp_size)
        elif is_pipeline_first_stage:
            labels = None
            loss_mask = None
            broadcast(tokens)
            broadcast(position_ids)
            if is_sft:
                cu_seqlens = receive_cu_seqlens()
                broadcast(max_seqlen)
                if cp_size > 1:
                    cu_seqlens_padded = receive_cu_seqlens()
            if create_attention_mask_in_dataloader:
                broadcast(attention_mask)
        elif is_pipeline_last_stage:
            tokens = None
            position_ids = None
            broadcast(labels)
            broadcast(loss_mask)
            if is_sft:
                cu_seqlens = receive_cu_seqlens()
                broadcast(max_seqlen)
                if cp_size > 1:
                    cu_seqlens_padded = receive_cu_seqlens()
            if create_attention_mask_in_dataloader:
                broadcast(attention_mask)
        elif is_sft:
            tokens = None
            labels = None
            loss_mask = None
            position_ids = None
            cu_seqlens = receive_cu_seqlens()
            broadcast(max_seqlen)
            if cp_size > 1:
                cu_seqlens_padded = receive_cu_seqlens()

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'cu_seqlens': cu_seqlens,
            'cu_seqlens_padded': cu_seqlens_padded,
            'max_seqlen': max_seqlen,
            'local_cp_size': local_cp_size,
            'hybrid_cp_group': None,
        }

    if tp_rank == 0 and batch.get('actual_seq_len') is None:
        raise AssertionError('reset-attention-mask requires actual_seq_len in the materialized batch.')
    actual_seq_len = broadcast_dynamic(
        batch['actual_seq_len'] if tp_rank == 0 else None,
        broadcast_src_rank,
        broadcast_group,
    )
    actual_seq_len = _validate_actual_seq_len(actual_seq_len)
    batch.pop('actual_seq_len', None)
    if _uses_p2p_eod_layout(args):
        actual_seq_len = _prepare_p2p_eod_actual_seq_len(actual_seq_len, batch, args)
    set_actual_seq_len(actual_seq_len)
    return batch
