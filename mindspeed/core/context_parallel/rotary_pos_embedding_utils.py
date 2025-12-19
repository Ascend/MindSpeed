# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch
from mindspeed.core.context_parallel import mpu as parallel_state
from mindspeed.core.context_parallel import get_args
from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
from mindspeed.utils import get_position_ids, generate_rearrange_idx_tensor
from mindspeed.core.context_parallel.model_parallel_utils import (get_context_parallel_for_hybrid_ulysses_world_size,
                                           get_context_parallel_for_hybrid_ulysses_rank,
                                           get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ring_rank)
from mindspeed.core.context_parallel.utils import get_remapped_seq_order


def get_pos_emb_on_this_cp_rank(pos_emb, seq_dim, cp_group):
    args = get_args()

    cp_expanded_by_2d_tp = args.tp_y > 1
    if args.context_parallel_algo == 'megatron_cp_algo':
        if args.attention_mask_type == 'general':
            pos_emb = _get_pos_emb_on_this_cp_rank_in_ulysses_cp(pos_emb, seq_dim)
        elif cp_expanded_by_2d_tp:
            pos_emb = _get_pos_emb_on_this_tp_y_cp_rank_in_megatron_cp(pos_emb, seq_dim)
        elif args.reset_position_ids and args.attention_mask_type == 'causal':
            return pos_emb
        else:
            pos_emb = _get_pos_emb_on_this_cp_rank_in_megatron_cp(pos_emb, seq_dim)
    elif args.context_parallel_algo == 'ulysses_cp_algo':
        if cp_expanded_by_2d_tp:
            pos_emb = _get_pos_emb_on_this_tp_y_cp_rank_in_ulysses_cp(pos_emb, seq_dim)
        else:
            pos_emb = _get_pos_emb_on_this_cp_rank_in_ulysses_cp(pos_emb, seq_dim)
    elif args.context_parallel_algo == 'hybrid_cp_algo':
        if args.attention_mask_type == 'general':
            pos_emb = _get_pos_emb_on_this_cp_rank_in_hybrid_cp_general(pos_emb, seq_dim)
        else:
            pos_emb = _get_pos_emb_on_this_cp_rank_in_hybrid_cp(pos_emb, seq_dim)
    elif args.context_parallel_algo == 'adaptive_cp_algo':
        pos_emb = _get_pos_emb_on_this_cp_rank_in_adaptive_cp(pos_emb, seq_dim)
    elif args.context_parallel_algo == 'hybrid_adaptive_cp_algo':
        pos_emb = _get_pos_emb_on_this_cp_rank_in_hybrid_adaptive_cp(pos_emb, seq_dim)
    return pos_emb


def _get_pos_emb_on_this_cp_rank_in_megatron_cp(pos_emb, seq_dim):
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_idx = torch.tensor(
        [cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True
    ).cuda(non_blocking=True)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], 2 * cp_size, -1, *pos_emb.shape[(seq_dim + 1):]
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2):])
    return pos_emb


def _get_pos_emb_on_this_tp_y_cp_rank_in_megatron_cp(pos_emb, seq_dim):
    origin_pos_emb_shape = pos_emb.shape
    tp_y_cp_group = TensorParallelYUnionCP()
    tp_y_cp_size = tp_y_cp_group.get_parallel_group_world_size()
    # [s, 1, 1, head_dim] ---> [2*tp_y_cp_size, s/(2*tp_y_cp_size), 1, 1, head_dim]
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], 2 * tp_y_cp_size, -1, *pos_emb.shape[(seq_dim + 1):]
    )
    rearrange_idx_tensor = generate_rearrange_idx_tensor(tp_y_cp_size)

    # Reorder pos embedding according dataset handling.
    # selected res shape: [2 * tp_y_cp_size, s / (2 * tp_y_cp_size), 1, 1, head_dim]
    pos_emb = pos_emb.index_select(seq_dim, index=rearrange_idx_tensor)
    pos_emb = pos_emb.view(*origin_pos_emb_shape)
    # viewed res shape: [tp_y_cp_sz, s/tp_y_cp_sz, 1, head_dim]
    pos_emb = pos_emb.view(
        *pos_emb.shape[0:seq_dim],
        tp_y_cp_size,
        pos_emb.shape[seq_dim] // tp_y_cp_size,
        *pos_emb.shape[(seq_dim + 1):],
    )
    # cur_rank_pos_emb shape: [s/cp, 1, 1, head_dim]
    tp_y_cp_rank = tp_y_cp_group.get_parallel_rank()
    cur_rank_pos_emb = pos_emb[tp_y_cp_rank].squeeze(axis=0)
    return cur_rank_pos_emb


def _get_pos_emb_on_this_cp_rank_in_ulysses_cp(pos_emb, seq_dim):
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    pos_emb = pos_emb.chunk(cp_size, dim=seq_dim)[cp_rank]

    return pos_emb


def _get_pos_emb_on_this_cp_rank_in_hybrid_cp(pos_emb, seq_dim):
    u_size = get_context_parallel_for_hybrid_ulysses_world_size()
    r_size = get_context_parallel_for_hybrid_ring_world_size()
    u_rank = get_context_parallel_for_hybrid_ulysses_rank()
    r_rank = get_context_parallel_for_hybrid_ring_rank()

    cp_idx = torch.tensor(
        [r_rank, (2 * r_size - r_rank - 1)], device="cpu", pin_memory=True
    ).cuda(non_blocking=True)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], 2 * r_size, -1, *pos_emb.shape[(seq_dim + 1):]
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2):])

    pos_emb = pos_emb.chunk(u_size, dim=seq_dim)[u_rank]

    return pos_emb


def _get_pos_emb_on_this_cp_rank_in_hybrid_cp_general(pos_emb, seq_dim):
    u_size = get_context_parallel_for_hybrid_ulysses_world_size()
    r_size = get_context_parallel_for_hybrid_ring_world_size()
    u_rank = get_context_parallel_for_hybrid_ulysses_rank()
    r_rank = get_context_parallel_for_hybrid_ring_rank()

    pos_emb = pos_emb.chunk(r_size, dim=seq_dim)[r_rank]
    pos_emb = pos_emb.chunk(u_size, dim=seq_dim)[u_rank]

    return pos_emb


def _get_pos_emb_on_this_cp_rank_in_adaptive_cp(pos_emd, seq_dim):
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()

    remapped_seq_order = get_remapped_seq_order()
    if remapped_seq_order is not None:
        per = pos_emd.shape[seq_dim] // cp_size
        index = torch.tensor(remapped_seq_order[cp_rank * per:(cp_rank + 1) * per], dtype=torch.int,
                             device=pos_emd.device)
        pos_emd = pos_emd.index_select(seq_dim, index)

    return pos_emd


def _get_pos_emb_on_this_cp_rank_in_hybrid_adaptive_cp(pos_emd, seq_dim):
    ulys_size = get_context_parallel_for_hybrid_ulysses_world_size()
    adap_size = get_context_parallel_for_hybrid_ring_world_size()
    ulys_rank = get_context_parallel_for_hybrid_ulysses_rank()
    adap_rank = get_context_parallel_for_hybrid_ring_rank()

    remapped_seq_order = get_remapped_seq_order()
    if remapped_seq_order is not None:
        per = pos_emd.shape[seq_dim] // adap_size // ulys_size
        which_per = adap_rank * ulys_size + ulys_rank
        index = torch.tensor(remapped_seq_order[which_per * per:(which_per + 1) * per], dtype=torch.int,
                             device=pos_emd.device)
        pos_emd = pos_emd.index_select(seq_dim, index)

    return pos_emd


def _get_pos_emb_on_this_tp_y_cp_rank_in_ulysses_cp(pos_emb, seq_dim):
    tp_y_cp_group = TensorParallelYUnionCP()
    tp_y_cp_size = tp_y_cp_group.get_parallel_group_world_size()

    cp_rank = tp_y_cp_group.get_parallel_rank()
    pos_emb = pos_emb.chunk(tp_y_cp_size, dim=seq_dim)[cp_rank]
    return pos_emb

