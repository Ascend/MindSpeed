# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import math
_HCCL_GROUP_BUFFER = {}


def parse_hccl_buffer_string(hccl_group_buffer):
    global _HCCL_GROUP_BUFFER

    if hccl_group_buffer is None:
        return

    allowed_keys = ["dp", "dp_cp", "cp", "mp", "mp_exp", "tp", "pp", "embd", "tp_dp_cp", 
                    "tp_dp", "tp_cp", "tp_exp", "exp", "dp_modulo_exp", "pp_new_stream", 
                    "cp2", "cp_ulysses", "cp_ring", "cp_ring_intra", "cp_ring_intra_overlap", "nd1_dim1", "ag_x_sd_rcv_overlap", 
                    "nd1_dim2", "ag_y_sd_rcv_overlap", "nd2_dim1", "nd2_dim2"]

    parts = hccl_group_buffer.split(';')
    for part in parts:
        key_value = part.split(':')
        if len(key_value) == 2:
            key = key_value[0].strip()
            value_str = key_value[1].strip()
            key = key.replace(' ', '')
            value_str = value_str.replace(' ', '')
            if key in allowed_keys:
                try:
                    value = int(value_str)
                    if value <= 0:
                        raise RuntimeError(f"Value {value} must be greater than 0")
                    _HCCL_GROUP_BUFFER[key] = value
                except ValueError as e:
                    raise RuntimeError(f"{value_str} is not a valid positive integer") from e
            else:
                raise RuntimeError(f"Key {key} is not allowed")
        else:
            raise RuntimeError("The str of hccl-group-buffer is not valid")


def hccl_buffer_auto_adaptive(args):
    seq_length = args.seq_length
    micro_batch_size = args.micro_batch_size
    hidden_size = args.hidden_size

    context_parallel_size = args.context_parallel_size
    tensor_model_parallel_size = args.tensor_model_parallel_size
    expert_model_parallel_size = args.expert_model_parallel_size

    moe_router_topk = args.moe_router_topk
    moe_token_dispatcher_type = args.moe_token_dispatcher_type

    context_parallel_algo = args.context_parallel_algo
    num_attention_heads = args.num_attention_heads
    group_query_attention = args.group_query_attention
    num_query_groups = args.num_query_groups

    global _HCCL_GROUP_BUFFER
    #The DP group, DP-CP group, and DP-EP group .Here, we take the default value of 200M.

    #Calculation of the maximum communication volume of the TP group.
    if moe_token_dispatcher_type is not None and moe_token_dispatcher_type == 'alltoall_seq':
        #No MOE + No SP, AllReduce MaxComm: S/CP * B * H * 2；No MOE + SP, AllGather MaxComm: S/CP * B * H
        hccl_tp_buffer_size_mlp = 2 * math.ceil(seq_length / context_parallel_size * micro_batch_size * hidden_size / 1024 / 1024)
        if args.sequence_parallel:
            _HCCL_GROUP_BUFFER['tp'] = hccl_tp_buffer_size_mlp
        else:
            _HCCL_GROUP_BUFFER['tp'] = hccl_tp_buffer_size_mlp * 2
        #MOE and AlltoAll_seq MaxComm: (S/CP/TP * B * H * topK).
        if args.hccl_ep_group_buffer_adaptive_factor > 0:
            hccl_tp_buffer_size_moe = 2 * math.ceil(args.hccl_ep_group_buffer_adaptive_factor * seq_length / context_parallel_size / tensor_model_parallel_size * micro_batch_size * hidden_size / 1024 / 1024 * moe_router_topk)
        else:
            hccl_tp_buffer_size_moe = 200
        _HCCL_GROUP_BUFFER['tp'] = max(hccl_tp_buffer_size_moe, _HCCL_GROUP_BUFFER['tp'])
    else:
        #MOE + SP, AllReduce MaxComm: S/CP * B * H * 2；No MOE + SP, AllGather MaxComm: S/CP * B * H
        hccl_tp_buffer_size_mlp = 2 * math.ceil(seq_length / context_parallel_size * micro_batch_size * hidden_size / 1024 / 1024)
        if args.sequence_parallel:
            _HCCL_GROUP_BUFFER['tp'] = hccl_tp_buffer_size_mlp
        else:
            _HCCL_GROUP_BUFFER['tp'] = hccl_tp_buffer_size_mlp * 2

    #Calculation of the maximum communication volume of the PP group.
    #P2P MaxComm::S/CP/TP * B *H
    if args.sequence_parallel:
        hccl_pp_buffer_size = 2 * math.ceil(seq_length / context_parallel_size / tensor_model_parallel_size * micro_batch_size * hidden_size / 1024 / 1024)
    else:
        hccl_pp_buffer_size = 2 * math.ceil(seq_length / context_parallel_size * micro_batch_size * hidden_size / 1024 / 1024)
    _HCCL_GROUP_BUFFER['pp'] = hccl_pp_buffer_size
    _HCCL_GROUP_BUFFER['pp_new_stream'] = hccl_pp_buffer_size

    #MP & MP-EXP groups for optimizer, based on num of zero gradients and max grad_norm. Just set a constant (default 10M).
    #It won't be used after the distributed optimizer is enabled.
    _HCCL_GROUP_BUFFER['mp'] = 10
    _HCCL_GROUP_BUFFER['mp_exp'] = 10

    #Calculation of the maximum communication volume of the EP group.
    #Moe of alltoall_seq, MaxComm:S/CP/TP * B * H * Topk
    if args.hccl_ep_group_buffer_adaptive_factor > 0: 
        hccl_ep_buffer_size = 2 * math.ceil(seq_length / context_parallel_size / tensor_model_parallel_size * micro_batch_size * hidden_size / 1024 / 1024 * moe_router_topk)
    else:
        hccl_ep_buffer_size = 200
    _HCCL_GROUP_BUFFER['exp'] = hccl_ep_buffer_size

    #Calculation of the maximum communication volume of the EP-TP group.
    #Moe of allgather, MaxComm:S/CP/TP * B * H * EP * TP
    #Moe of alltoall_seq + moe-tp-extend-ep , MaxComm:S/CP/TP * B * H * topK
    if moe_token_dispatcher_type is not None and moe_token_dispatcher_type == 'allgather': 
        if args.hccl_ep_group_buffer_adaptive_factor > 0:
            hccl_tp_ep_buffer_size = 2 * math.ceil(args.hccl_ep_group_buffer_adaptive_factor * seq_length / context_parallel_size * micro_batch_size * hidden_size * expert_model_parallel_size / 1024 / 1024)
        else:
            hccl_tp_ep_buffer_size = 200
        _HCCL_GROUP_BUFFER['tp_exp'] = hccl_ep_buffer_size
    elif moe_token_dispatcher_type is not None and moe_token_dispatcher_type == 'alltoall_seq' and args.moe_tp_extend_ep:
        if args.hccl_ep_group_buffer_adaptive_factor > 0:
            hccl_tp_ep_buffer_size = 2 * math.ceil(args.hccl_ep_group_buffer_adaptive_factor * seq_length / context_parallel_size / tensor_model_parallel_size * micro_batch_size * hidden_size * moe_router_topk / 1024 / 1024)
        else:
            hccl_tp_ep_buffer_size = 200
        _HCCL_GROUP_BUFFER['tp_exp'] = hccl_ep_buffer_size

    #TP-CP group in 8.0 for seq count by experts & Router bal_loss. Small comm vol, set const (default 10M).
    _HCCL_GROUP_BUFFER['tp_cp'] = 10

    #Calculation of the maximum communication volume of the CP、CP2、CP_Ring、CP_Ulysess group.
    #CP of RingAttention，SendRecv，MaxComm:S/CP * B * (H / headcount * GQA /TP ) * 2
    #CP of Ulysess，All2All，MaxComm:S/CP * B * (H / TP)
    #CP_ulysess & CP_ring like CP in max comm. CP2 is half of CP.
    if context_parallel_algo == 'ulysses_cp_algo' or context_parallel_algo is None:
        hccl_cp_buffer_size = 2 * math.ceil(seq_length / context_parallel_size * micro_batch_size * hidden_size / tensor_model_parallel_size / 1024 / 1024)
        _HCCL_GROUP_BUFFER['cp'] = hccl_cp_buffer_size
    elif context_parallel_algo == 'megatron_cp_algo':
        if group_query_attention:
            hccl_cp2_buffer_size = 2 * math.ceil(
                seq_length / context_parallel_size * micro_batch_size * hidden_size / num_attention_heads * num_query_groups / tensor_model_parallel_size / 1024 / 1024)
            hccl_cp_buffer_size = 2 * 2 * math.ceil(
                seq_length / context_parallel_size * micro_batch_size * hidden_size / num_attention_heads * num_query_groups / tensor_model_parallel_size / 1024 / 1024)
        else:
            hccl_cp2_buffer_size = 2 * math.ceil(
                seq_length / context_parallel_size * micro_batch_size * hidden_size / num_attention_heads / tensor_model_parallel_size / 1024 / 1024)
            hccl_cp_buffer_size = 2 * 2 * math.ceil(
                seq_length / context_parallel_size * micro_batch_size * hidden_size / num_attention_heads / tensor_model_parallel_size / 1024 / 1024)

        if args.cp_window_size > 1:
            if args.use_cp_send_recv_overlap:
                _HCCL_GROUP_BUFFER['cp2'] = hccl_cp2_buffer_size
                _HCCL_GROUP_BUFFER['cp'] = hccl_cp2_buffer_size
                _HCCL_GROUP_BUFFER['cp_ring_intra'] = hccl_cp2_buffer_size
                _HCCL_GROUP_BUFFER['cp_ring_intra_overlap'] = hccl_cp2_buffer_size
            else:
                _HCCL_GROUP_BUFFER['cp'] = hccl_cp_buffer_size
                _HCCL_GROUP_BUFFER['cp_ring_intra'] = hccl_cp_buffer_size
        else:
            if args.use_cp_send_recv_overlap:
                _HCCL_GROUP_BUFFER['cp2'] = hccl_cp2_buffer_size
                _HCCL_GROUP_BUFFER['cp'] = hccl_cp2_buffer_size
            else:
                _HCCL_GROUP_BUFFER['cp'] = hccl_cp_buffer_size
    elif context_parallel_algo == 'hybrid_cp_algo':
        ulysses_context_parallel_size = args.ulysses_degree_in_cp
        ring_context_parallel_size = context_parallel_size / ulysses_context_parallel_size
        hccl_cp_ulysess_buffer_size = 2 * math.ceil(seq_length / ulysses_context_parallel_size * micro_batch_size * hidden_size / tensor_model_parallel_size / 1024 / 1024)
        if group_query_attention:
            hccl_cp_ring_buffer_size = 2 * math.ceil(
                seq_length / ring_context_parallel_size * micro_batch_size * hidden_size / num_attention_heads * num_query_groups / tensor_model_parallel_size / 1024 / 1024)
        else:
            hccl_cp_ring_buffer_size = 2 * math.ceil(
                seq_length / ring_context_parallel_size * micro_batch_size * hidden_size / num_attention_heads / tensor_model_parallel_size / 1024 / 1024)
        if args.cp_window_size > 1:
            if args.use_cp_send_recv_overlap:
                _HCCL_GROUP_BUFFER['cp_ulysses'] = hccl_cp_ulysess_buffer_size
                _HCCL_GROUP_BUFFER['cp_ring'] = hccl_cp_ring_buffer_size
                _HCCL_GROUP_BUFFER['cp2'] = hccl_cp_ring_buffer_size
                _HCCL_GROUP_BUFFER['cp_ring_intra'] = hccl_cp_ring_buffer_size
                _HCCL_GROUP_BUFFER['cp_ring_intra_overlap'] = hccl_cp_ring_buffer_size
                #The CP group is used to calculate losses. The traffic volume is very small and is given a fixed value of 10M.
                _HCCL_GROUP_BUFFER['cp'] = 10
            else:
                _HCCL_GROUP_BUFFER['cp_ulysses'] = hccl_cp_ulysess_buffer_size
                _HCCL_GROUP_BUFFER['cp_ring'] = hccl_cp_ring_buffer_size * 2
                _HCCL_GROUP_BUFFER['cp_ring_intra'] = hccl_cp_ring_buffer_size * 2
                #The CP group is used to calculate losses. The traffic volume is very small and is given a fixed value of 10M.
                _HCCL_GROUP_BUFFER['cp'] = 10
        else:
            if args.use_cp_send_recv_overlap:
                _HCCL_GROUP_BUFFER['cp_ulysses'] = hccl_cp_ulysess_buffer_size
                _HCCL_GROUP_BUFFER['cp_ring'] = hccl_cp_ring_buffer_size
                _HCCL_GROUP_BUFFER['cp2'] = hccl_cp_ring_buffer_size
                #The CP group is used to calculate losses. The traffic volume is very small and is given a fixed value of 10M.
                _HCCL_GROUP_BUFFER['cp'] = 10
            else:
                _HCCL_GROUP_BUFFER['cp_ulysses'] = hccl_cp_ulysess_buffer_size
                _HCCL_GROUP_BUFFER['cp_ring'] = hccl_cp_ring_buffer_size * 2
                #The CP group is used to calculate losses. The traffic volume is very small and is given a fixed value of 10M.
                _HCCL_GROUP_BUFFER['cp'] = 10
