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


def broadcast_dynamic(item):
    if item is not None:
        item = item.npu()
        item_len = torch.tensor(item.numel(), device=torch.cuda.current_device())
        _broadcast(item_len)
        _broadcast(item)
    else:
        item_len = torch.empty((), dtype=torch.int64, device=torch.cuda.current_device())
        _broadcast(item_len)
        item = torch.empty([item_len.item()], dtype=torch.int64, device=torch.cuda.current_device())
        _broadcast(item)
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


def get_batch_on_this_tp_rank(data_iterator, mtp_on_this_rank: bool = False):
    """Build the EOD batch without installing a MindSpeed CP slicing path."""
    args = get_args()

    if mpu.get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator) if data_iterator is not None else None
        batch = {
            'tokens': data['tokens'].cuda(non_blocking=True),
            'labels': data['labels'].cuda(non_blocking=True),
            'loss_mask': data['loss_mask'].cuda(non_blocking=True),
            'attention_mask': None if 'attention_mask' not in data else data['attention_mask'].cuda(non_blocking=True),
            'position_ids': data['position_ids'].cuda(non_blocking=True),
        }
        if args.pipeline_model_parallel_size == 1 or mtp_on_this_rank:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
        elif mpu.is_pipeline_last_stage():
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            if args.reset_attention_mask:
                _broadcast(batch['position_ids'])
        elif args.reset_attention_mask:
            _broadcast(batch['position_ids'])
        if args.reset_attention_mask:
            actual_seq_len = _validate_actual_seq_len(broadcast_dynamic(data['actual_seq_len']))
            if _uses_p2p_eod_layout(args):
                actual_seq_len = _prepare_p2p_eod_actual_seq_len(actual_seq_len, batch, args)
            set_actual_seq_len(actual_seq_len)
    else:
        tokens = torch.empty(
            (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device()
        )
        labels = torch.empty(
            (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device()
        )
        loss_mask = torch.empty(
            (args.micro_batch_size, args.seq_length), dtype=torch.float32, device=torch.cuda.current_device()
        )
        attention_mask = (
            torch.empty(
                (args.micro_batch_size, 1, args.seq_length, args.seq_length),
                dtype=torch.bool,
                device=torch.cuda.current_device(),
            )
            if getattr(args, 'create_attention_mask_in_dataloader', False)
            else None
        )
        position_ids = torch.empty(
            (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device()
        )
        if args.pipeline_model_parallel_size == 1 or mtp_on_this_rank:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)
        elif mpu.is_pipeline_first_stage():
            labels = loss_mask = None
            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)
        elif mpu.is_pipeline_last_stage():
            tokens = None
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            if args.reset_attention_mask:
                _broadcast(position_ids)
            else:
                position_ids = None
        elif args.reset_attention_mask:
            _broadcast(position_ids)
        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }
        if args.reset_attention_mask:
            actual_seq_len = _validate_actual_seq_len(broadcast_dynamic(None))
            if _uses_p2p_eod_layout(args):
                actual_seq_len = _prepare_p2p_eod_actual_seq_len(actual_seq_len, batch, args)
            set_actual_seq_len(actual_seq_len)
    return batch
