# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Expert parallel groups."""
import os
from functools import wraps
from typing import Optional
from datetime import timedelta

import torch
import torch_npu
import megatron

from mindspeed.core.simple_parallel_cfg import SimpleParallelCfg
from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP

_CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP = None
_CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES = None
_CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING = None
_PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM = None

_CONTEXT_PARALLEL_RANKS_FOR_HYBRID_ULYSSES = None
_CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING = None

_CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW = None
_CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV = None
_CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV = None
_CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW = None
_CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW_SEND_RECV_OVERLAP = None

_TP_X_PARALLEL_RING_RANKS = None
_TP_Y_PARALLEL_RING_RANKS = None

_TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1 = None
_TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2 = None
_TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM1 = None
_TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM2 = None
_TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM1 = None
_TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM2 = None
_TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM1 = None
_TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM2 = None
_TP_X_SD_RCV_OVERLAP_GROUP = None
_TP_Y_SD_RCV_OVERLAP_GROUP = None
_TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1_RANK = None
_TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2_RANK = None
_TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1_WORLD_SIZE = None
_TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2_WORLD_SIZE = None

_TENSOR_AND_CONTEXT_PARALLEL_GROUP = None
_TENSOR_AND_CONTEXT_PARALLEL_GLOBAL_RANKS = None

_HCCL_GROUP_BUFFER = None


def parse_hccl_buffer_string(hccl_group_buffer):
    global _HCCL_GROUP_BUFFER

    if hccl_group_buffer == None:
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
                except ValueError:
                    raise RuntimeError(f"{value_str} is not a valid positive integer")
            else:
                raise RuntimeError(f"Key {key} is not allowed")
        else:
            raise RuntimeError("The str of hccl-group-buffer is not valid")


def hccl_buffer_auto_adaptive():
    import math
    from megatron.training import get_args
    args = get_args()

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

    global _HCCL_GROUP_BUFFER
    #The DP group, DP-CP group, and DP-EP group .Here, we take the default value of 200M.

    #Calculation of the maximum communication volume of the TP group.
    if moe_token_dispatcher_type is not None and moe_token_dispatcher_type == 'alltoall':
        #No MOE + No SP, AllReduce MaxComm: S/CP * B * H * 2；No MOE + SP, AllGather MaxComm: S/CP * B * H
        hccl_tp_buffer_size_mlp = 2 * math.ceil(seq_length / context_parallel_size * micro_batch_size * hidden_size / 1024 / 1024)
        if args.sequence_parallel:
            _HCCL_GROUP_BUFFER['tp'] = hccl_tp_buffer_size_mlp
        else:
            _HCCL_GROUP_BUFFER['tp'] = hccl_tp_buffer_size_mlp * 2
        #MOE and AlltoAll MaxComm: (S/CP/TP * B * H * topK).
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
    #Moe of alltoall, MaxComm:S/CP/TP * B * H * Topk
    if args.hccl_ep_group_buffer_adaptive_factor > 0: 
        hccl_ep_buffer_size = 2 * math.ceil(seq_length / context_parallel_size / tensor_model_parallel_size * micro_batch_size * hidden_size / 1024 / 1024 * moe_router_topk)
    else:
        hccl_ep_buffer_size = 200
    _HCCL_GROUP_BUFFER['exp'] = hccl_ep_buffer_size

    #Calculation of the maximum communication volume of the EP-TP group.
    #Moe of allgather, MaxComm:S/CP/TP * B * H * EP * TP
    #Moe of alltoall + moe-tp-extend-ep , MaxComm:S/CP/TP * B * H * topK
    if moe_token_dispatcher_type is not None and moe_token_dispatcher_type == 'allgather': 
        if args.hccl_ep_group_buffer_adaptive_factor > 0:
            hccl_tp_ep_buffer_size = 2 * math.ceil(args.hccl_ep_group_buffer_adaptive_factor * seq_length / context_parallel_size * micro_batch_size * hidden_size * expert_model_parallel_size / 1024 / 1024)
        else:
            hccl_tp_ep_buffer_size = 200
        _HCCL_GROUP_BUFFER['tp_exp'] = hccl_ep_buffer_size
    elif moe_token_dispatcher_type is not None and moe_token_dispatcher_type == 'alltoall' and args.moe_tp_extend_ep:
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
    elif context_parallel_algo == 'megatron_cp_algo' :
        hccl_cp2_buffer_size = 2 * math.ceil(seq_length / context_parallel_size * micro_batch_size * hidden_size / num_attention_heads * group_query_attention / tensor_model_parallel_size / 1024 / 1024)
        hccl_cp_buffer_size = 2 * 2 * math.ceil(seq_length / context_parallel_size * micro_batch_size * hidden_size / num_attention_heads * group_query_attention / tensor_model_parallel_size / 1024 / 1024)
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
        hccl_cp_ring_buffer_size = 2 * math.ceil(seq_length / ring_context_parallel_size * micro_batch_size * hidden_size / num_attention_heads * group_query_attention / tensor_model_parallel_size / 1024 / 1024)
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
        

def get_nccl_options_wrapper(get_nccl_options):
    @wraps(get_nccl_options)
    def wrapper(pg_name, nccl_comm_cfgs):
        from megatron.training import get_args
        args = get_args()
        if args.hccl_group_buffer is not None or args.hccl_group_buffer_adaptive:
            global _HCCL_GROUP_BUFFER
            if _HCCL_GROUP_BUFFER.get(pg_name) is not None:
                options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
                options.hccl_config = {"hccl_buffer_size":_HCCL_GROUP_BUFFER[pg_name]}
                return options
        return get_nccl_options(pg_name, nccl_comm_cfgs)
    return wrapper


def initialize_model_parallel_wrapper(initialize_model_parallel):
    @wraps(initialize_model_parallel)
    def wrapper(
            tensor_model_parallel_size: int = 1,
            pipeline_model_parallel_size: int = 1,
            virtual_pipeline_model_parallel_size: Optional[int] = None,
            pipeline_model_parallel_split_rank: Optional[int] = None,
            use_sharp: bool = False,
            context_parallel_size: int = 1,
            expert_model_parallel_size: int = 1,
            nccl_communicator_config_path: Optional[str] = None,
            distributed_timeout_minutes: int = 30,
            order: str = "tp-cp-ep-dp-pp",
    ):
        from megatron.training.utils import print_rank_0
        from megatron.training import get_args
        args = get_args()

        global _HCCL_GROUP_BUFFER
        _HCCL_GROUP_BUFFER = {}

        if args.hccl_group_buffer_adaptive:
            hccl_buffer_auto_adaptive()
            print_rank_0(f"hccl_group_buffer_adaptive: {_HCCL_GROUP_BUFFER}")

        if args.hccl_group_buffer is not None:
            parse_hccl_buffer_string(args.hccl_group_buffer)

        if virtual_pipeline_model_parallel_size is not None:
            megatron.core.parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
            megatron.core.parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

        # Megatron doesn't allow ep & cp combination, set ep to 1 to bypass that, ep related groups will be regenerated
        initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            None,
            pipeline_model_parallel_split_rank,
            use_sharp,
            context_parallel_size,
            1,
            nccl_communicator_config_path,
            distributed_timeout_minutes,
            order
        )

        rank = torch.distributed.get_rank()
        world_size: int = torch.distributed.get_world_size()
        num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
        num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
        data_parallel_size: int = world_size // (
                tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
        )

        if data_parallel_size * context_parallel_size % expert_model_parallel_size != 0:
            raise RuntimeError(
                f"data_parallel_size * context_parallel_size ({data_parallel_size * context_parallel_size}) is not "
                f"divisible by expert_model_parallel_size "
            )

        nccl_comm_cfgs = {}
        if nccl_communicator_config_path is not None:
            import yaml

            with open(nccl_communicator_config_path, "r") as stream:
                nccl_comm_cfgs = yaml.safe_load(stream)

        all_data_parallel_group_ranks = []
        all_data_parallel_group_ranks_with_cp = []
        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups
            for j in range(context_parallel_size * tensor_model_parallel_size):
                ranks = range(
                    start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size
                )
                all_data_parallel_group_ranks.append(list(ranks))
            for j in range(tensor_model_parallel_size):
                ranks_with_cp = range(
                    start_rank + j, end_rank, tensor_model_parallel_size
                )
                all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))

        timeout = timedelta(minutes=distributed_timeout_minutes)

        # Regenerate ep related groups because ep is set to 1 in initialize_model_parallel func
        tensor_and_data_group_size_with_cp: int = (tensor_model_parallel_size * data_parallel_size
                                                   * context_parallel_size)
        num_tensor_and_data_groups_with_cp: int = world_size // tensor_and_data_group_size_with_cp
        tensor_and_expert_group_size: int = tensor_model_parallel_size * expert_model_parallel_size
        num_expert_groups: int = data_parallel_size * context_parallel_size // expert_model_parallel_size
        all_tensor_and_expert_group_ranks = []
        for i in range(num_tensor_and_data_groups_with_cp):
            for j in range(num_expert_groups):
                start_rank = i * tensor_and_data_group_size_with_cp + j * tensor_and_expert_group_size
                end_rank = i * tensor_and_data_group_size_with_cp + (j + 1) * tensor_and_expert_group_size
                ranks = range(start_rank, end_rank)
                all_tensor_and_expert_group_ranks.append(list(ranks))
                group = torch.distributed.new_group(
                    ranks, timeout=timeout,
                    pg_options=megatron.core.parallel_state.get_nccl_options('tp_exp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    megatron.core.parallel_state._TENSOR_AND_EXPERT_PARALLEL_GROUP = group

        all_dp_modulo_exp_group_ranks = []
        for i in range(num_tensor_and_data_groups_with_cp):
            start_rank = i * tensor_and_data_group_size_with_cp
            end_rank = (i + 1) * tensor_and_data_group_size_with_cp
            for j in range(tensor_and_expert_group_size):
                ranks = range(start_rank + j, end_rank, tensor_and_expert_group_size)
                all_dp_modulo_exp_group_ranks.append(list(ranks))
                group = torch.distributed.new_group(
                    ranks, timeout=timeout,
                    pg_options=megatron.core.parallel_state.get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)
                )
                group_gloo = torch.distributed.new_group(ranks, backend="gloo")
                if rank in ranks:
                    megatron.core.parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP = group
                    megatron.core.parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo

        # Build expert parallel groups

        all_ep_groups = []
        for dp_cp_ranks in all_data_parallel_group_ranks_with_cp:
            for i in range(0, len(dp_cp_ranks), expert_model_parallel_size):
                ranks = dp_cp_ranks[i:i + expert_model_parallel_size]
                all_ep_groups.append(list(ranks))
                group = torch.distributed.new_group(
                    ranks, pg_options=megatron.core.parallel_state.get_nccl_options('exp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    megatron.core.parallel_state._EXPERT_MODEL_PARALLEL_GROUP = group

        all_tp_groups = []
        for i in range(num_tensor_model_parallel_groups):
            ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
            all_tp_groups.append(list(ranks))
        
        initialize_context_parallel_group_for_send_recv_overlap(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            context_parallel_size,
            nccl_comm_cfgs
        )

        initialize_context_parallel_group_for_hybrid_cp(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            context_parallel_size,
            nccl_comm_cfgs
        )

        initialize_context_parallel_group_for_double_ring(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            context_parallel_size,
            nccl_comm_cfgs
        )


        print_rank_0(f"all tp gourps {all_tp_groups}")
        print_rank_0(f"all ep groups {all_ep_groups}")
        print_rank_0(f"all dp groups {all_data_parallel_group_ranks}")
        print_rank_0(f"all_dp_modulo_exp_group_ranks {all_dp_modulo_exp_group_ranks}")
        print_rank_0(f"all_tensor_and_expert_group_ranks {all_tensor_and_expert_group_ranks}")
        print_rank_0(f"all_data_parallel_group_ranks_with_cp {all_data_parallel_group_ranks_with_cp}")

        global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM
        if _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM is not None:
            raise AttributeError('Pipeline parallel group for new stream is already initialized')
        num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
        for i in range(num_pipeline_model_parallel_groups):
            ranks = range(i, world_size, num_pipeline_model_parallel_groups)
            group = torch.distributed.new_group(
                ranks, pg_options=megatron.core.parallel_state.get_nccl_options('pp_new_stream', nccl_comm_cfgs)
            )
            if rank in ranks:
                _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM = group

        from megatron.training import get_args
        args = get_args()
        nd1_dim1_sz = args.nd1_dim1_size if args.use_nd_matmul else args.tp_x
        nd2_dim1_sz = args.nd2_dim1_size if args.use_nd_matmul else args.tp_y
        initialize_ndmm_parallel_group(
            nccl_comm_cfgs,
            tensor_model_parallel_size=tensor_model_parallel_size,
            nd1_dim1_size=nd1_dim1_sz,
            nd2_dim1_size=nd2_dim1_sz,
        )
        if args.tp_2d:
            tp_y_cp_group = TensorParallelYUnionCP(
                parallel_cfg=SimpleParallelCfg(
                    dp=data_parallel_size,
                    pp=pipeline_model_parallel_size,
                    tp=tensor_model_parallel_size,
                    cp=context_parallel_size,
                    ep=expert_model_parallel_size,
                    tp_x=get_args().tp_x,
                    tp_y=get_args().tp_y,
                ),
                pg_name="tp-y-cp",
                overlap_gp_name="tp-y-cp-overlap",
                nccl_comm_cfgs=nccl_comm_cfgs
            )
            print(f'tp_y_cp_group.global_ranks={tp_y_cp_group.global_ranks} for rank {rank}')
    return wrapper


def get_ring_group_for_intra_window():
    global _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW
    return _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW


def get_ring_group_for_intra_window_send_recv_overlap():
    global _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW_SEND_RECV_OVERLAP
    return _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW_SEND_RECV_OVERLAP


def get_ring_ranks_for_intra_window():
    global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW
    assert _CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW is not None
    return _CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW


def get_ring_ranks_for_inter_window_kv():
    global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV
    assert _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV is not None
    return _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV


def get_ring_ranks_for_inter_window_dkv():
    global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV
    assert _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV is not None
    return _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV


def initialize_context_parallel_group_for_send_recv_overlap(
    tensor_model_parallel_size,
    pipeline_model_parallel_size,
    context_parallel_size,
    nccl_comm_cfgs
):
    from megatron.training import get_args
    if not get_args().use_cp_send_recv_overlap:
        return
    if get_args().tp_2d and get_args().tp_y > 1:
        return
    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    data_parallel_size: int = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )
    global _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP
    for i in range(pipeline_model_parallel_size):
        for j in range(data_parallel_size):
            start_rank = (
                    i * num_pipeline_model_parallel_groups
                    + j * tensor_model_parallel_size * context_parallel_size
            )
            end_rank = (
                    i * num_pipeline_model_parallel_groups
                    + (j + 1) * tensor_model_parallel_size * context_parallel_size
            )
            for k in range(tensor_model_parallel_size):
                ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
                group_send_recv_overlap = torch.distributed.new_group(
                    ranks, pg_options=megatron.core.parallel_state.get_nccl_options('cp2', nccl_comm_cfgs)
                )
                if rank in ranks:
                    _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP = group_send_recv_overlap


def initialize_context_parallel_group_for_hybrid_cp(
    tensor_model_parallel_size,
    pipeline_model_parallel_size,
    context_parallel_size,
    nccl_comm_cfgs
):
    from megatron.training import get_args
    if (not hasattr(get_args(), 'context_parallel_algo') or
            (get_args().context_parallel_algo != 'hybrid_cp_algo' and get_args().context_parallel_algo != 'hybrid_adaptive_cp_algo')):
        return

    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    data_parallel_size: int = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )

    ulysses_degree = get_args().ulysses_degree_in_cp
    assert (context_parallel_size > ulysses_degree and context_parallel_size % ulysses_degree == 0)
    ring_degree = context_parallel_size // ulysses_degree

    global _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES
    global _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_ULYSSES
    global _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING
    global _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING
    for i in range(pipeline_model_parallel_size):
        for j in range(data_parallel_size):
            start_rank = (
                    i * num_pipeline_model_parallel_groups
                    + j * tensor_model_parallel_size * context_parallel_size
            )
            end_rank = (
                    i * num_pipeline_model_parallel_groups
                    + (j + 1) * tensor_model_parallel_size * context_parallel_size
            )
            for k in range(tensor_model_parallel_size):
                # cp ranks
                ranks = list(range(start_rank + k, end_rank, tensor_model_parallel_size))
                # ulysses cp ranks. 
                # Ulysses need higher communication bandwidth than Ring.
                # Try to put Ulysses ranks in the same node.
                for m in range(ring_degree):
                    ulysses_ranks = [ranks[idx] for idx in range(m * ulysses_degree, (m + 1) * ulysses_degree)]
                    ulysses_group = torch.distributed.new_group(
                        ulysses_ranks,
                        pg_options=megatron.core.parallel_state.get_nccl_options('cp_ulysses', nccl_comm_cfgs)
                    )
                    if rank in ulysses_ranks:
                        _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES = ulysses_group
                        _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_ULYSSES = ulysses_ranks

                # ring cp ranks
                for m in range(ulysses_degree):
                    ring_ranks = [ranks[idx] for idx in range(m, len(ranks), ulysses_degree)]
                    ring_group = torch.distributed.new_group(
                        ring_ranks, pg_options=megatron.core.parallel_state.get_nccl_options('cp_ring', nccl_comm_cfgs)
                    )
                    if rank in ring_ranks:
                        _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING = ring_group
                        _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING = ring_ranks


def initialize_context_parallel_group_for_double_ring(
    tensor_model_parallel_size,
    pipeline_model_parallel_size,
    context_parallel_size,
    nccl_comm_cfgs,
):
    from megatron.training import get_args
    args = get_args()
    if args.tp_2d:
        return
    if context_parallel_size == 1 or args.context_parallel_algo not in ['megatron_cp_algo', 'hybrid_cp_algo']:
        return
        
    use_hybrid_cp = args.context_parallel_algo == 'hybrid_cp_algo' and args.ulysses_degree_in_cp > 1

    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    data_parallel_size: int = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )

    def _initialize_helper(
        rank,
        ring_global_ranks,
        window_size
    ):
        from megatron.training import get_args
        import megatron.core.parallel_state as ps
        global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW
        global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV
        global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV
        global _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW
        global _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW_SEND_RECV_OVERLAP

        ring_size = len(ring_global_ranks)
        inter_size = ring_size // window_size
        for wid in range(inter_size):
            intra_ranks = [ring_global_ranks[idx] for idx in range(wid * window_size, (wid + 1) * window_size)]
            intra_group = torch.distributed.new_group(intra_ranks, pg_options=ps.get_nccl_options('cp_ring_intra', nccl_comm_cfgs))
            intra_group_for_send_recv_overlap = None
            if args.use_cp_send_recv_overlap:
                intra_group_for_send_recv_overlap = torch.distributed.new_group(intra_ranks, pg_options=ps.get_nccl_options('cp_ring_intra_overlap', nccl_comm_cfgs))

            if rank in intra_ranks:
                _CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW = intra_ranks
                _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW = intra_group
                _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW_SEND_RECV_OVERLAP = intra_group_for_send_recv_overlap

        for inner_id in range(window_size):
            inter_ranks = [ring_global_ranks[idx] for idx in range(inner_id, ring_size, window_size)]
            if rank in inter_ranks:
                _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV = inter_ranks
                break

        for inner_id in range(window_size):
            inter_dkv_ranks = []
            cur_rank = ring_global_ranks[inner_id]
            cur_idx = inner_id
            cur_window = 0
            while cur_rank not in inter_dkv_ranks:
                inter_dkv_ranks.append(cur_rank)
                cur_window = (cur_window + 1) % inter_size
                window_start = cur_window * window_size
                cur_idx = window_start + (cur_idx + 1) % window_size
                cur_rank = ring_global_ranks[cur_idx]

            if rank in inter_dkv_ranks:
                _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV = inter_dkv_ranks
                break


    for i in range(pipeline_model_parallel_size):
        for j in range(data_parallel_size):
            start_rank = (
                    i * num_pipeline_model_parallel_groups
                    + j * tensor_model_parallel_size * context_parallel_size
            )
            end_rank = (
                    i * num_pipeline_model_parallel_groups
                    + (j + 1) * tensor_model_parallel_size * context_parallel_size
            )
            for k in range(tensor_model_parallel_size):
                cp_ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)

                if use_hybrid_cp:
                    ulysses_degree = get_args().ulysses_degree_in_cp
                    assert (context_parallel_size > ulysses_degree and context_parallel_size % ulysses_degree == 0)
                    # ring cp ranks
                    for m in range(ulysses_degree):
                        ring_ranks = [cp_ranks[idx] for idx in range(m, len(cp_ranks), ulysses_degree)]

                        _initialize_helper(rank, ring_ranks, args.cp_window_size)
                else:
                    _initialize_helper(rank, cp_ranks, args.cp_window_size)


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-cp-ep-dp-pp",
) -> None:
    """Initialize model data parallel groups.

    Args:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        pipeline_model_parallel_split_rank (int, optional):
            For models with both an encoder and decoder, the rank in
            pipeline to switch between encoder and decoder (i.e. the
            first rank of the decoder). This allows the user to set
            the pipeline parallel size of the encoder and decoder
            independently. For example, if
            pipeline_model_parallel_size is 8 and
            pipeline_model_parallel_split_rank is 3, then ranks 0-2
            will be the encoder and ranks 3-7 will be the decoder.

        use_sharp (bool, default = False):
            Set the use of SHARP for the collective communications of
            data-parallel process groups. When `True`, run barrier
            within each data-parallel process group, which specifies
            the SHARP application target groups.

        context_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a context parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a context parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and context parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do context parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Context parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a context parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of context parallelism on data parallel group for
            weight gradient all-reduce.

        expert_model_parallel_size (int, default = 1):
            The number of Mixture of Experts parallel GPUs in each expert
            parallel group.

        nccl_communicator_config_path (str, default = None):
            Path to the yaml file of NCCL communicator configurations.
            `min_ctas`, `max_ctas`, and `cga_cluster_size` can be set
            for each communicator.

        distributed_timeout_minutes (int, default = 30): Timeout, in
            minutes,for operations executed against distributed
            process groups. See PyTorch documentation at
            https://pytorch.org/docs/stable/distributed.html for
            caveats.

        order (str, default=tp-dp-pp):
            The rank initialization order of parallelism. Now we support
            tp-dp-pp and tp-pp-dp orders.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

    """
    from datetime import timedelta
    import megatron.core.parallel_state as ps
    from megatron.training import get_args
    args = get_args()

    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    if (
        world_size
        % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)
        != 0
    ):
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
            f"x context_parallel_size ({context_parallel_size})"
        )

    data_parallel_size: int = world_size // (
        tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )

    if data_parallel_size % expert_model_parallel_size != 0:
        raise RuntimeError(
            f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
        )

    if expert_model_parallel_size > 1 and context_parallel_size > 1:
        raise RuntimeError(
            f"combination of expert model prallellism and context parallelism is not supported"
        )

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
            )
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        ps._PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    rank_generator = ps.RankGenerator(
        tp=tensor_model_parallel_size,
        ep=expert_model_parallel_size,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order=order,
    )
    timeout = timedelta(minutes=distributed_timeout_minutes)

    # Build the data-parallel groups.
    assert ps._DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'

    for ranks in rank_generator.get_ranks('dp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('dp', nccl_comm_cfgs)
        )
        group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")
        if rank in ranks:
            ps._DATA_PARALLEL_GROUP = group
            ps._DATA_PARALLEL_GROUP_GLOO = group_gloo
            ps._DATA_PARALLEL_GLOBAL_RANKS = ranks
    for ranks_with_cp in rank_generator.get_ranks('dp-cp'):
        group_with_cp = torch.distributed.new_group(
            ranks_with_cp, timeout=timeout, pg_options=ps.get_nccl_options('dp_cp', nccl_comm_cfgs)
        )
        group_with_cp_gloo = torch.distributed.new_group(
            ranks_with_cp, timeout=timeout, backend="gloo"
        )
        if rank in ranks_with_cp:
            ps._DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
            ps._DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
            ps._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

    # Apply SHARP to DP process groups
    if use_sharp:
        if rank == 0:
            print(
                "The number of process groups to use SHARP with depends on the type "
                "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                "process groups and QM2 supports up to 256 process groups. We apply "
                "SHARP to the communications of the data-parallel domain. If the "
                "number of data-parallel process groups is larger than the max "
                "process groups that the network switch supports, the communication "
                "will fall back to non-SHARP operators. To enable SHARP, "
                "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
            )
        torch.distributed.barrier(
            group=ps.get_data_parallel_group(with_context_parallel=True),
            device_ids=[torch.cuda.current_device()],
        )
        # Set `NCCL_COLLNET_ENABLE=0` to restrict SHARP application to DP process groups
        os.environ["NCCL_COLLNET_ENABLE"] = "0"

    # Build the context-parallel groups.
    global _TENSOR_AND_CONTEXT_PARALLEL_GROUP
    global _TENSOR_AND_CONTEXT_PARALLEL_GLOBAL_RANKS
    assert ps._CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'
    assert _TENSOR_AND_CONTEXT_PARALLEL_GROUP is None, 'tensor and context parallel group is already initialized'
    for ranks in rank_generator.get_ranks('cp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            ps._CONTEXT_PARALLEL_GROUP = group
            ps._CONTEXT_PARALLEL_GLOBAL_RANKS = ranks

    for ranks in rank_generator.get_ranks('tp-cp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('tp_cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _TENSOR_AND_CONTEXT_PARALLEL_GROUP = group
            _TENSOR_AND_CONTEXT_PARALLEL_GLOBAL_RANKS = ranks


    # Build the model-parallel groups.
    assert ps._MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'
    for ranks in rank_generator.get_ranks('tp-pp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('mp', nccl_comm_cfgs)
        )
        if rank in ranks:
            ps._MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    assert (
        ps._TENSOR_MODEL_PARALLEL_GROUP is None
    ), 'tensor model parallel group is already initialized'
    for ranks in rank_generator.get_ranks('tp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('tp', nccl_comm_cfgs)
        )
        if rank in ranks:
            ps._TENSOR_MODEL_PARALLEL_GROUP = group
            ps._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    assert (
        ps._PIPELINE_MODEL_PARALLEL_GROUP is None
    ), 'pipeline model parallel group is already initialized'
    assert ps._EMBEDDING_GROUP is None, 'embedding group is already initialized'
    assert ps._POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'
    for ranks in rank_generator.get_ranks('pp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('pp', nccl_comm_cfgs)
        )
        if rank in ranks:
            ps._PIPELINE_MODEL_PARALLEL_GROUP = group
            ps._PIPELINE_GLOBAL_RANKS = ranks
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            if pipeline_model_parallel_split_rank is not None:
                if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                    embedding_ranks = [
                        ranks[0],
                        ranks[pipeline_model_parallel_split_rank],
                        ranks[-1],
                    ]
                if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = torch.distributed.new_group(
            embedding_ranks, timeout=timeout, pg_options=ps.get_nccl_options('embd', nccl_comm_cfgs)
        )
        if rank in embedding_ranks:
            ps._EMBEDDING_GROUP = group
        if rank in ranks:
            ps._EMBEDDING_GLOBAL_RANKS = embedding_ranks

        group = torch.distributed.new_group(
            position_embedding_ranks,
            timeout=timeout,
            pg_options=ps.get_nccl_options('embd', nccl_comm_cfgs),
        )
        if rank in position_embedding_ranks:
            ps._POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            ps._POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    # Build the tensor + data parallel groups.
    assert (
        ps._TENSOR_AND_DATA_PARALLEL_GROUP is None
    ), 'Tensor + data parallel group is already initialized'
    for ranks in rank_generator.get_ranks('tp-dp-cp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('tp_dp_cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            ps._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group
    for ranks in rank_generator.get_ranks('tp-dp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('tp_dp', nccl_comm_cfgs)
        )
        if rank in ranks:
            ps._TENSOR_AND_DATA_PARALLEL_GROUP = group

    # Build the tensor + expert parallel groups
    assert ps._EXPERT_MODEL_PARALLEL_GROUP is None, 'Expert parallel group is already initialized'
    assert (
        ps._TENSOR_AND_EXPERT_PARALLEL_GROUP is None
    ), 'Tensor + expert parallel group is already initialized'
    assert (
        ps._DATA_MODULO_EXPERT_PARALLEL_GROUP is None
    ), 'Data modulo expert group is already initialized'
    for ranks in rank_generator.get_ranks('tp-ep', independent_ep=True):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('tp_exp', nccl_comm_cfgs)
        )
        if rank in ranks:
            ps._TENSOR_AND_EXPERT_PARALLEL_GROUP = group

    for ranks in rank_generator.get_ranks('ep', independent_ep=True):
        group = torch.distributed.new_group(
            ranks, pg_options=ps.get_nccl_options('exp', nccl_comm_cfgs)
        )
        if rank in ranks:
            ps._EXPERT_MODEL_PARALLEL_GROUP = group

    for ranks in rank_generator.get_ranks('dp', independent_ep=True):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)
        )
        group_gloo = torch.distributed.new_group(ranks, backend="gloo")
        if rank in ranks:
            ps._DATA_MODULO_EXPERT_PARALLEL_GROUP = group
            ps._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    ps._set_global_memory_buffer()
    

def get_context_parallel_group_for_send_recv_overlap(check_initialized=True):
    """Get the context parallel group for send-recv overlap the caller rank belongs to."""
    if check_initialized:
        assert (
            _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP is not None
        ), 'context parallel group for send-recv overlap is not initialized'
    return _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP


def get_context_parallel_next_rank():
    """Return the global rank that follows the caller in the context parallel"""
    import megatron.core.parallel_state as ps
    assert ps._CONTEXT_PARALLEL_GLOBAL_RANKS is not None, "Context parallel group is not initialized"
    rank_in_context = ps.get_context_parallel_rank()
    world_size = ps.get_context_parallel_world_size()
    return ps._CONTEXT_PARALLEL_GLOBAL_RANKS[(rank_in_context + 1) % world_size]


def get_context_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the context parallel"""
    import megatron.core.parallel_state as ps
    assert ps._CONTEXT_PARALLEL_GLOBAL_RANKS is not None, "Context parallel group is not initialized"
    rank_in_context = ps.get_context_parallel_rank()
    world_size = ps.get_context_parallel_world_size()
    return ps._CONTEXT_PARALLEL_GLOBAL_RANKS[(rank_in_context - 1) % world_size]


def get_pipeline_parallel_group_for_new_stream():
    if _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM is None:
        raise AttributeError('Pipeline parallel group of backward is not initialized')
    return _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM


def get_context_parallel_group_for_hybrid_ulysses(check_initialized=True):
    """Get the context parallel group for hybrid ulysses the caller rank belongs to."""
    if check_initialized:
        assert (
            _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES is not None
        ), 'context parallel group for hybrid ulysses is not initialized'
    return _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES


def get_context_parallel_for_hybrid_ulysses_world_size():
    return torch.distributed.get_world_size(group=get_context_parallel_group_for_hybrid_ulysses())


def get_context_parallel_for_hybrid_ulysses_rank():
    return torch.distributed.get_rank(group=get_context_parallel_group_for_hybrid_ulysses())


def get_context_parallel_group_for_hybrid_ring(check_initialized=True):
    """Get the context parallel group for hybrid ring the caller rank belongs to."""
    if check_initialized:
        assert (
            _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING is not None
        ), 'context parallel group for hybrid ring is not initialized'
    return _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING


def get_context_parallel_for_hybrid_ring_world_size():
    return torch.distributed.get_world_size(group=get_context_parallel_group_for_hybrid_ring())


def get_context_parallel_for_hybrid_ring_rank():
    return torch.distributed.get_rank(group=get_context_parallel_group_for_hybrid_ring())


def get_context_parallel_for_hybrid_ring_global_ranks():
    assert (_CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING is not None
            ), 'context parallel group for hybrid ring is not initialized'
    global _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING
    return _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING


def get_tp_x_ring_global_ranks():
    global _TP_X_PARALLEL_RING_RANKS
    assert (_TP_X_PARALLEL_RING_RANKS is not None), 'TP-X parallel group for ring is not initialized'
    return _TP_X_PARALLEL_RING_RANKS


def get_tp_y_ring_global_ranks():
    global _TP_Y_PARALLEL_RING_RANKS
    assert (_TP_Y_PARALLEL_RING_RANKS is not None), 'TP-Y parallel group for ring is not initialized'
    return _TP_Y_PARALLEL_RING_RANKS


def destroy_model_parallel_wrapper(destroy_model_parallel):
    @wraps(destroy_model_parallel)
    def wrapper():
        destroy_model_parallel()

        global _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP
        global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM
        global _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING
        global _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES
        global _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING
        global _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_ULYSSES
        global _TP_X_PARALLEL_RING_RANKS
        global _TP_Y_PARALLEL_RING_RANKS
        global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1
        global _TP_X_SD_RCV_OVERLAP_GROUP
        global _TP_Y_SD_RCV_OVERLAP_GROUP
        global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2
        global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1_RANK
        global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2_RANK
        global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1_WORLD_SIZE
        global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2_WORLD_SIZE
        global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM1
        global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM2
        global _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM1
        global _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM2
        global _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM1
        global _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM2
        global _TENSOR_AND_CONTEXT_PARALLEL_GROUP
        global _TENSOR_AND_CONTEXT_PARALLEL_GLOBAL_RANKS
        _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP = None
        _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM = None
        _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING = None
        _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES = None
        _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING = None
        _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_ULYSSES = None
        _TENSOR_AND_CONTEXT_PARALLEL_GROUP = None
        _TENSOR_AND_CONTEXT_PARALLEL_GLOBAL_RANKS = None
        _TP_X_PARALLEL_RING_RANKS = None
        _TP_Y_PARALLEL_RING_RANKS = None
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1 = None
        _TP_X_SD_RCV_OVERLAP_GROUP = None
        _TP_Y_SD_RCV_OVERLAP_GROUP = None
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2 = None
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1_RANK = None
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2_RANK = None
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1_WORLD_SIZE = None
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2_WORLD_SIZE = None
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM1 = None
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM2 = None
        _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM1 = None
        _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM2 = None
        _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM1 = None
        _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM2 = None

    return wrapper


def get_tensor_model_parallel_group_for_nd1_dim1(check_initialized=True):
    if check_initialized and _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1 is None:
        raise AssertionError('tensor model parallel group for nd1 dim1 is not initialized')
    return _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1


def get_tp_x_sd_rcv_overlap_group(check_initialized=True):
    if check_initialized and _TP_X_SD_RCV_OVERLAP_GROUP is None:
        raise AssertionError('tp-x send recv overlap group is not initialized')
    return _TP_X_SD_RCV_OVERLAP_GROUP


def get_tp_y_sd_rcv_overlap_group(check_initialized=True):
    if check_initialized and _TP_Y_SD_RCV_OVERLAP_GROUP is None:
        raise AssertionError('tp-y send recv overlap group is not initialized')
    return _TP_Y_SD_RCV_OVERLAP_GROUP


def get_tensor_model_parallel_group_for_nd1_dim2(check_initialized=True):
    if check_initialized and _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2 is None:
        raise AssertionError('tensor model parallel group for nd1 dim2 is not initialized')
    return _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2


def get_tensor_model_parallel_group_for_nd2_dim1(check_initialized=True):
    if check_initialized and _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM1 is None:
        raise AssertionError('tensor model parallel group for nd2 dim1 is not initialized')
    return _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM1


def get_tensor_model_parallel_group_for_nd1_dim1_rank():
    global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1_RANK
    if _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1_RANK is None:
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1_RANK = torch.distributed.get_rank(
            group=get_tensor_model_parallel_group_for_nd1_dim1())

    return _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1_RANK


def get_tensor_model_parallel_group_for_nd1_dim2_rank():
    global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2_RANK
    if _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2_RANK is None:
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2_RANK = torch.distributed.get_rank(
            group=get_tensor_model_parallel_group_for_nd1_dim2())

    return _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2_RANK


def get_tensor_model_parallel_group_for_nd1_dim1_world_size():
    global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1_WORLD_SIZE
    if _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1_WORLD_SIZE is None:
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1_WORLD_SIZE = torch.distributed.get_world_size(
            group=get_tensor_model_parallel_group_for_nd1_dim1())

    return _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1_WORLD_SIZE


def get_tensor_model_parallel_group_for_nd1_dim2_world_size():
    global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2_WORLD_SIZE
    if _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2_WORLD_SIZE is None:
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2_WORLD_SIZE = torch.distributed.get_world_size(
            group=get_tensor_model_parallel_group_for_nd1_dim2())

    return _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2_WORLD_SIZE


def get_tensor_model_parallel_group_for_nd2_dim2(check_initialized=True):
    if check_initialized and _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM2 is None:
        raise AssertionError('tensor model parallel group for nd2 dim2 is not initialized')
    return _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM2


def get_tensor_model_parallel_world_size_for_nd1_dim1():
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM1
    if _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM1 is None:
        _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM1 = torch.distributed.get_world_size(
            group=get_tensor_model_parallel_group_for_nd1_dim1()
        )
    return _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM1


def get_tensor_model_parallel_world_size_for_nd1_dim2():
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM2
    if _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM2 is None:
        _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM2 = torch.distributed.get_world_size(
            group=get_tensor_model_parallel_group_for_nd1_dim2()
        )
    return _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM2


def get_tensor_model_parallel_world_size_for_nd2_dim1():
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM1
    if _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM1 is None:
        _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM1 = torch.distributed.get_world_size(
            group=get_tensor_model_parallel_group_for_nd2_dim1()
        )
    return _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM1


def get_tensor_model_parallel_world_size_for_nd2_dim2():
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM2
    if _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM2 is None:
        _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM2 = torch.distributed.get_world_size(
            group=get_tensor_model_parallel_group_for_nd2_dim2()
        )
    return _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM2


def initialize_ndmm_parallel_group(
    nccl_comm_cfgs: dict,
    tensor_model_parallel_size: int = 1,
    nd1_dim1_size: int = 1,
    nd2_dim1_size: int = 1,
) -> None:
    import megatron.core.parallel_state as ps
    from megatron.training import get_args
    from megatron.training.global_vars import _ensure_var_is_not_initialized

    args = get_args()
    if not (args.use_nd_matmul or args.tp_2d):
        return

    global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1
    _ensure_var_is_not_initialized(
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1, 'nd1_dim1'
    )

    global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2
    _ensure_var_is_not_initialized(
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2, 'nd1_dim2'
    )

    global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM1
    _ensure_var_is_not_initialized(
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM1, 'nd2_dim1'
    )

    global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM2
    _ensure_var_is_not_initialized(
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM2, 'nd2_dim2'
    )

    global _TP_X_PARALLEL_RING_RANKS
    _ensure_var_is_not_initialized(_TP_X_PARALLEL_RING_RANKS, 'tp_x_ring_ranks')

    global _TP_Y_PARALLEL_RING_RANKS
    _ensure_var_is_not_initialized(_TP_Y_PARALLEL_RING_RANKS, 'tp_y_ring_ranks')

    global _TP_X_SD_RCV_OVERLAP_GROUP
    _ensure_var_is_not_initialized(_TP_X_SD_RCV_OVERLAP_GROUP, 'tp_x_overlap_ranks')

    global _TP_Y_SD_RCV_OVERLAP_GROUP
    _ensure_var_is_not_initialized(_TP_Y_SD_RCV_OVERLAP_GROUP, 'tp_y_overlap_ranks')

    if tensor_model_parallel_size % nd1_dim1_size != 0:
        raise RuntimeError(
            f"tensor_model_parallel_size can't divisible by nd1_dim1_size"
        )

    if tensor_model_parallel_size % nd2_dim1_size != 0:
        raise RuntimeError(
            f"tensor_model_parallel_size can't divisible by nd2_dim1_size"
        )

    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    num_tensor_model_parallel_group: int = world_size // tensor_model_parallel_size

    tp_nd1_dim1_groups = []
    tp_nd1_dim2_groups = []
    tp_nd2_dim1_groups = []
    tp_nd2_dim2_groups = []
    for i in range(num_tensor_model_parallel_group):
        for j in range(tensor_model_parallel_size // nd1_dim1_size):
            ranks = range(
                i * tensor_model_parallel_size + j * nd1_dim1_size,
                i * tensor_model_parallel_size + (j + 1) * nd1_dim1_size
            )
            tp_nd1_dim1_groups.append(list(ranks))
            group = torch.distributed.new_group(
                ranks, pg_options=ps.get_nccl_options('nd1_dim1', nccl_comm_cfgs)
            )
            if args.enable_overlap_ag_with_matmul or args.enable_backward_overlap_ag_with_matmul:
                tp_x_ag_overlap_group = torch.distributed.new_group(
                    ranks, pg_options=ps.get_nccl_options('ag_x_sd_rcv_overlap', nccl_comm_cfgs)
                )
            else:
                tp_x_ag_overlap_group = None
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1 = group
                _TP_X_SD_RCV_OVERLAP_GROUP = tp_x_ag_overlap_group
                _TP_X_PARALLEL_RING_RANKS = ranks

        nd1_dim2_size = tensor_model_parallel_size // nd1_dim1_size
        for j in range(tensor_model_parallel_size // nd1_dim2_size):
            ranks = range(
                i * tensor_model_parallel_size + j,
                (i + 1) * tensor_model_parallel_size,
                nd1_dim1_size
            )
            tp_nd1_dim2_groups.append(list(ranks))
            group = torch.distributed.new_group(
                ranks, pg_options=ps.get_nccl_options('nd1_dim2', nccl_comm_cfgs)
            )
            if args.enable_overlap_ag_with_matmul or args.enable_backward_overlap_ag_with_matmul:
                tp_y_ag_overlap_group = torch.distributed.new_group(
                    ranks, pg_options=ps.get_nccl_options('ag_y_sd_rcv_overlap', nccl_comm_cfgs)
                )
            else:
                tp_y_ag_overlap_group = None
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2 = group
                _TP_Y_SD_RCV_OVERLAP_GROUP = tp_y_ag_overlap_group
                _TP_Y_PARALLEL_RING_RANKS = ranks

        for j in range(tensor_model_parallel_size // nd2_dim1_size):
            ranks = range(
                i * tensor_model_parallel_size + j * nd2_dim1_size,
                i * tensor_model_parallel_size + (j + 1) * nd2_dim1_size
            )
            tp_nd2_dim1_groups.append(list(ranks))
            group = torch.distributed.new_group(
                ranks, pg_options=ps.get_nccl_options('nd2_dim1', nccl_comm_cfgs)
            )
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM1 = group

        nd2_dim2_size = tensor_model_parallel_size // nd2_dim1_size
        for j in range(tensor_model_parallel_size // nd2_dim2_size):
            ranks = range(
                i * tensor_model_parallel_size + j,
                (i + 1) * tensor_model_parallel_size,
                nd2_dim1_size
            )
            tp_nd2_dim2_groups.append(list(ranks))
            group = torch.distributed.new_group(
                ranks, pg_options=ps.get_nccl_options('nd2_dim2', nccl_comm_cfgs)
            )
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM2 = group


def get_tensor_and_context_parallel_group(check_initialized=True):
    """Get the context parallel group the caller rank belongs to."""
    if check_initialized:
        assert _TENSOR_AND_CONTEXT_PARALLEL_GROUP is not None, 'tensor model and context parallel group is not initialized'
    return _TENSOR_AND_CONTEXT_PARALLEL_GROUP


def get_tensor_and_context_parallel_world_size():
    """Return world size for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(group=get_tensor_and_context_parallel_group())
    else:
        return 0


def get_tensor_and_context_parallel_rank():
    """Return my rank for the tensor and context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_tensor_and_context_parallel_group())
    else:
        return 0


def get_data_parallel_group_gloo_replace(with_context_parallel=False):
    """Get the data parallel group-gloo the caller rank belongs to."""
    import megatron.core.parallel_state as ps

    if with_context_parallel:
        assert (
            ps._DATA_PARALLEL_GROUP_WITH_CP_GLOO is None
        ), 'data parallel group-gloo with context parallel combined should be None when args.disable_gloo_group is True'
        return ps._DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert ps._DATA_PARALLEL_GROUP_GLOO is None, 'data parallel group-gloo should be None when args.disable_gloo_group is True'
        return ps._DATA_PARALLEL_GROUP


def get_data_modulo_expert_parallel_group_gloo_replace():
    import megatron.core.parallel_state as ps

    assert (
        ps._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO is None
    ), 'data modulo expert parallel group-gloo should be None when args.disable_gloo_group is True'
    return ps._DATA_MODULO_EXPERT_PARALLEL_GROUP


def new_group_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        from megatron.training import get_args
        if get_args().disable_gloo_group:
            if "backend" in kwargs and kwargs["backend"] == "gloo":
                return None
        return fn(*args, **kwargs)
    return wrapper