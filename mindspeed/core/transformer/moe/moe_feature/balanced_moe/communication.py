# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
import torch_npu
from megatron.training import get_args

from mindspeed.core.transformer.moe.moe_feature.balanced_moe.parallel_state import (
    get_expert_model_parallel_global_ranks,
    get_hot_expert_group_list)
from mindspeed.core.transformer.moe.moe_feature.balanced_moe.modules.moe_layer import get_shared_grad_for_hot_experts


def _groupedmlp_hot_expert_params_broadcast(local_experts, hot_experts_list, hot_experts, params):
    num_local_experts = params[0]
    num_hot_experts = len(hot_experts_list)
    local_expert_indices = params[1]

    local_w1 = local_experts.weight1.view(num_local_experts, local_experts.config.hidden_size, -1).contiguous()
    local_w2 = local_experts.weight2.view(num_local_experts, -1, local_experts.config.hidden_size).contiguous()
    hot_experts_weight1_view = hot_experts.weight1.view(num_hot_experts, hot_experts.config.hidden_size, -1)
    hot_experts_weight2_view = hot_experts.weight2.view(num_hot_experts, -1, hot_experts.config.hidden_size)
    src_rank_list = []
    data_ptr_list = []

    global_ranks = get_expert_model_parallel_global_ranks()

    for offset, hot_expert_id in enumerate(hot_experts_list):
        # Broadcast
        src_rank_idx = hot_expert_id // num_local_experts
        src_rank = global_ranks[src_rank_idx]
        # Determine if the current process is the sender or receiver
        if hot_expert_id in local_expert_indices:
            # Sender
            local_pos = local_expert_indices.index(hot_expert_id)
            b_w1 = local_w1[local_pos]
            b_w2 = local_w2[local_pos]
        else:
            # Receiver
            b_w1 = hot_experts_weight1_view[offset]
            b_w2 = hot_experts_weight2_view[offset]

        src_rank_list.append(src_rank)
        data_ptr_list.append([b_w1, b_w2])

    _multiple_broadcast(src_rank_list, data_ptr_list, num_hot_experts, params)


def _multiple_broadcast(src_rank_list, data_ptr_list, num_hot_experts, params):
    hot_expert_broadcast_handles = params[4]

    args = get_args()
    hot_expert_group_list_len = args.trans_hot_expert_group_num
    hot_expert_group_list = get_hot_expert_group_list()

    for weight_ofst in range(2):
        for offset in range(num_hot_experts):
            src_rank = src_rank_list[offset]
            weight = data_ptr_list[offset][weight_ofst]
            comm_group = hot_expert_group_list[src_rank % hot_expert_group_list_len]
            handle = torch.distributed.broadcast(weight.data,
                                                 src_rank,
                                                 comm_group,
                                                 async_op=True)
            hot_expert_broadcast_handles[offset].append(handle)


def _groupedmlp_hot_expert_gradient_reduce(hot_experts_list, hot_experts, params):
    num_local_experts, _, expert_broadcast_streams, hot_expert_finish_events, _, hot_expert_inter_ep_grad_reduce_handles = params
    num_hot_experts = len(hot_experts_list)
    hidden_size = hot_experts.config.hidden_size
    grad_hot_w1, grad_hot_w2 = get_shared_grad_for_hot_experts()

    grad_hot_w1.copy_(
        hot_experts.weight1.grad.view(num_hot_experts, hidden_size, -1),
    )
    grad_hot_w2.copy_(
        hot_experts.weight2.grad.view(num_hot_experts, -1, hidden_size),
    )

    tgt_rank_list = []
    data_ptr_list = []

    args = get_args()
    hot_expert_group_list_len = args.trans_hot_expert_group_num
    hot_expert_group_list = get_hot_expert_group_list()
    global_ranks = get_expert_model_parallel_global_ranks()
    for offset, hot_expert_id in enumerate(hot_experts_list):
        # Reduce
        dst_rank_idx = hot_expert_id // num_local_experts
        dst_rank = global_ranks[dst_rank_idx]
        tgt_rank_list.append(dst_rank)
        # Sender & Receiver
        data_ptr_list.append((
            grad_hot_w1[offset].contiguous(),
            grad_hot_w2[offset].contiguous()
        ))

    for weight_ofst in range(2):
        for offset in range(num_hot_experts):
            if weight_ofst == 0:
                expert_broadcast_streams[offset].wait_event(hot_expert_finish_events[offset])

            tgt_rank = tgt_rank_list[offset]
            weight = data_ptr_list[offset][weight_ofst]
            comm_group = hot_expert_group_list[tgt_rank % hot_expert_group_list_len]

            handle = torch.distributed.reduce(weight,
                                              tgt_rank,
                                              group=comm_group,
                                              async_op=True)
            hot_expert_inter_ep_grad_reduce_handles[weight_ofst][offset] = handle
