# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

import functools

import torch
from megatron.core import mpu
from mindspeed.core.parallel_state import (get_context_parallel_for_hybrid_ulysses_world_size,
                                             get_context_parallel_for_hybrid_ulysses_rank,
                                             get_context_parallel_for_hybrid_ring_world_size,
                                             get_context_parallel_for_hybrid_ring_rank)


@functools.lru_cache(4096)
def print_rank_0_once(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def get_batch_on_this_cp_rank(batch):
    """ Slice batch input along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
    """

    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    from megatron.training import get_args

    args = get_args()
    cp_size = args.context_parallel_size
    if not cp_size > 1:
        return batch

    if args.context_parallel_algo == 'megatron_cp_algo':
        if args.cp_attention_mask_type == 'general':
            batch = _get_batch_on_this_cp_rank_in_megatron_cp_general(batch)
        else:
            batch = _get_batch_on_this_cp_rank_in_megatron_cp(batch)
    elif args.context_parallel_algo == 'ulysses_cp_algo':
        batch = _get_batch_on_this_cp_rank_in_ulysses_cp(batch)
    elif args.context_parallel_algo == 'hybrid_cp_algo':
        if args.cp_attention_mask_type == 'general':
            batch = _get_batch_on_this_cp_rank_in_hybrid_cp_general(batch)
        else:
            batch = _get_batch_on_this_cp_rank_in_hybrid_cp(batch)
    return batch


def _get_batch_on_this_cp_rank_in_megatron_cp(batch):
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    for key, val in batch.items():
        if key == 'attention_mask':
            continue
        if val is not None:
            seq_dim = 1 if key != 'attention_mask' else 2
            val = val.view(
                *val.shape[0:seq_dim],
                2 * cp_size,
                val.shape[seq_dim] // (2 * cp_size),
                *val.shape[(seq_dim + 1):],
            )
            index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=val.device)
            val = val.index_select(seq_dim, index)
            val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
            batch[key] = val

    return batch


def _get_batch_on_this_cp_rank_in_megatron_cp_general(batch):
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    for key, val in batch.items():
        if key == 'attention_mask' and val is not None:
            if len(val.shape) != 2:
                raise AssertionError("The fusion attention operator currently only support 2D attention mask.")
            seq_dim = 0
            mask_row = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
            mask_list = [m.contiguous() for m in mask_row.chunk(cp_size, dim=1)]
            batch[key] = mask_list
            continue
        if val is not None:
            seq_dim = 1
            val = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
            batch[key] = val
        
    return batch


def _get_batch_on_this_cp_rank_in_ulysses_cp(batch):
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    for key, val in batch.items():
        if key == 'attention_mask':
            continue
        if val is not None:
            seq_dim = 1 if key != 'attention_mask' else 2
            val = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
            batch[key] = val

    return batch


def _get_batch_on_this_cp_rank_in_hybrid_cp(batch):
    u_size = get_context_parallel_for_hybrid_ulysses_world_size()
    r_size = get_context_parallel_for_hybrid_ring_world_size()

    u_rank = get_context_parallel_for_hybrid_ulysses_rank()
    r_rank = get_context_parallel_for_hybrid_ring_rank()

    for key, val in batch.items():
        if key == 'attention_mask':
            continue
        if val is not None:
            seq_dim = 1 if key != 'attention_mask' else 2
            val = val.view(
                *val.shape[0:seq_dim],
                2 * r_size,
                val.shape[seq_dim] // (2 * r_size),
                *val.shape[(seq_dim + 1):],
            )
            index = torch.tensor([r_rank, (2 * r_size - r_rank - 1)], device=val.device)
            val = val.index_select(seq_dim, index)
            val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
            val = val.chunk(u_size, dim=seq_dim)[u_rank].contiguous()
            batch[key] = val

    return batch


def _get_batch_on_this_cp_rank_in_hybrid_cp_general(batch):
    u_size = get_context_parallel_for_hybrid_ulysses_world_size()
    r_size = get_context_parallel_for_hybrid_ring_world_size()

    u_rank = get_context_parallel_for_hybrid_ulysses_rank()
    r_rank = get_context_parallel_for_hybrid_ring_rank()

    for key, val in batch.items():
        if key == 'attention_mask' and val is not None:
            if len(val.shape) != 2:
                raise AssertionError("The fusion attention operator currently only support 2D attention mask.")
            seq_dim = 0
            mask_row = val.chunk(r_size, dim=seq_dim)[r_rank].contiguous()
            mask_list = [m.contiguous() for m in mask_row.chunk(r_size, dim=1)]
            batch[key] = mask_list
            continue

        if val is not None:
            seq_dim = 1
            val = val.chunk(r_size, dim=seq_dim)[r_rank].contiguous()
            val = val.chunk(u_size, dim=seq_dim)[u_rank].contiguous()
            batch[key] = val

    return batch
