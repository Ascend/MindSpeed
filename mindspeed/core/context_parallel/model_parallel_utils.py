# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from functools import wraps
from typing import Optional, Callable, List
import torch
from mindspeed.core.context_parallel import get_args, get_nccl_options

_CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP = None
_CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES = None
_CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING = None

_CONTEXT_PARALLEL_RANKS_FOR_HYBRID_ULYSSES = None
_CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING = None

_CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW = None
_CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV = None
_CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV = None
_CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW = None
_CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW_SEND_RECV_OVERLAP = None


def initialize_context_parallel_group_for_send_recv_overlap(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        context_parallel_size,
        nccl_comm_cfgs
):
    if not get_args().use_cp_send_recv_overlap:
        return
    # when tp_y > 1, use TensorParallelYUnionCP
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
                    ranks, pg_options=get_nccl_options('cp2', nccl_comm_cfgs)
                )
                if rank in ranks:
                    _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP = group_send_recv_overlap


def initialize_context_parallel_group_for_hybrid_cp(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        context_parallel_size,
        nccl_comm_cfgs
):
    if (not hasattr(get_args(), 'context_parallel_algo') or (get_args().context_parallel_algo != 'hybrid_cp_algo' and get_args().context_parallel_algo != 'hybrid_adaptive_cp_algo')):
        return

    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    data_parallel_size: int = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )

    ulysses_degree = get_args().ulysses_degree_in_cp
    if not (context_parallel_size > ulysses_degree and context_parallel_size % ulysses_degree == 0):
        raise AssertionError('Invalid ulysses degree configuration')
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
                        pg_options=get_nccl_options('cp_ulysses', nccl_comm_cfgs)
                    )
                    if rank in ulysses_ranks:
                        _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES = ulysses_group
                        _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_ULYSSES = ulysses_ranks

                # ring cp ranks
                for m in range(ulysses_degree):
                    ring_ranks = [ranks[idx] for idx in range(m, len(ranks), ulysses_degree)]
                    ring_group = torch.distributed.new_group(
                        ring_ranks, pg_options=get_nccl_options('cp_ring', nccl_comm_cfgs)
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
        global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW
        global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV
        global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV
        global _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW
        global _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW_SEND_RECV_OVERLAP

        ring_size = len(ring_global_ranks)
        inter_size = ring_size // window_size
        for wid in range(inter_size):
            intra_ranks = [ring_global_ranks[idx] for idx in range(wid * window_size, (wid + 1) * window_size)]
            intra_group = torch.distributed.new_group(intra_ranks, pg_options=get_nccl_options('cp_ring_intra', nccl_comm_cfgs))
            intra_group_for_send_recv_overlap = None
            if args.use_cp_send_recv_overlap:
                intra_group_for_send_recv_overlap = torch.distributed.new_group(intra_ranks, pg_options=get_nccl_options('cp_ring_intra_overlap', nccl_comm_cfgs))

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
                    if not (context_parallel_size > ulysses_degree and context_parallel_size % ulysses_degree == 0):
                        raise AssertionError('Invalid ulysses degree configuration')
                    # ring cp ranks
                    for m in range(ulysses_degree):
                        ring_ranks = [cp_ranks[idx] for idx in range(m, len(cp_ranks), ulysses_degree)]

                        _initialize_helper(rank, ring_ranks, args.cp_window_size)
                else:
                    _initialize_helper(rank, cp_ranks, args.cp_window_size)


def get_context_parallel_group_for_send_recv_overlap(check_initialized=True):
    """Get the context parallel group for send-recv overlap the caller rank belongs to."""
    if check_initialized:
        if _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP is None:
            raise AssertionError('context parallel group for send-recv overlap is not initialized')
    return _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP


def initialize_model_parallel_cp_wrapper(initialize_model_parallel):
    @wraps(initialize_model_parallel)
    def wrapper(
            tensor_model_parallel_size: int = 1,
            pipeline_model_parallel_size: int = 1,
            virtual_pipeline_model_parallel_size: Optional[int] = None,
            pipeline_model_parallel_comm_backend: Optional[str] = None,
            use_sharp: bool = False,
            context_parallel_size: int = 1,
            hierarchical_context_parallel_sizes: Optional[List[int]] = None,
            expert_model_parallel_size: int = 1,
            num_distributed_optimizer_instances: int = 1,
            expert_tensor_parallel_size: Optional[int] = None,
            nccl_communicator_config_path: Optional[str] = None,
            distributed_timeout_minutes: int = 30,
            order: str = "tp-cp-ep-dp-pp",
            get_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
            get_position_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
            create_gloo_process_groups: bool = True,
            high_priority_stream_groups: Optional[List[str]] = None,
            sharp_enabled_group: Optional[str] = None,
    ):
        initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_comm_backend,
            use_sharp,
            context_parallel_size,
            hierarchical_context_parallel_sizes,
            expert_model_parallel_size,
            num_distributed_optimizer_instances,
            expert_tensor_parallel_size,
            nccl_communicator_config_path,
            distributed_timeout_minutes,
            order,
            get_embedding_ranks,
            get_position_embedding_ranks,
            create_gloo_process_groups,
            high_priority_stream_groups,
            sharp_enabled_group
        )
        nccl_comm_cfgs = {}
        if nccl_communicator_config_path is not None:
            import yaml
            with open(nccl_communicator_config_path, "r") as stream:
                nccl_comm_cfgs = yaml.safe_load(stream)

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

    return wrapper


def destroy_model_parallel_cp_wrapper(destroy_model_parallel):
    @wraps(destroy_model_parallel)
    def wrapper():
        destroy_model_parallel()
        global _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP
        global _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING
        global _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES
        global _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING
        global _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_ULYSSES

        _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP = None
        _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING = None
        _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES = None
        _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING = None
        _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_ULYSSES = None

    return wrapper


def get_context_parallel_group_for_hybrid_ulysses(check_initialized=True):
    """Get the context parallel group for hybrid ulysses the caller rank belongs to."""
    if check_initialized:
        if _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES is None:
            raise AssertionError('context parallel group for hybrid ulysses is not initialized')
    return _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES


def get_context_parallel_for_hybrid_ulysses_world_size():
    return torch.distributed.get_world_size(group=get_context_parallel_group_for_hybrid_ulysses())


def get_context_parallel_for_hybrid_ulysses_rank():
    return torch.distributed.get_rank(group=get_context_parallel_group_for_hybrid_ulysses())


def get_context_parallel_group_for_hybrid_ring(check_initialized=True):
    """Get the context parallel group for hybrid ring the caller rank belongs to."""
    if check_initialized:
        if _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING is None:
            raise AssertionError('context parallel group for hybrid ring is not initialized')
    return _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING


def get_context_parallel_for_hybrid_ring_world_size():
    return torch.distributed.get_world_size(group=get_context_parallel_group_for_hybrid_ring())


def get_context_parallel_for_hybrid_ring_rank():
    return torch.distributed.get_rank(group=get_context_parallel_group_for_hybrid_ring())


def get_context_parallel_for_hybrid_ring_global_ranks():
    if _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING is None:
        raise AssertionError('context parallel group for hybrid ring is not initialized')
    global _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING
    return _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING


def get_ring_group_for_intra_window():
    global _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW
    return _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW


def get_ring_group_for_intra_window_send_recv_overlap():
    global _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW_SEND_RECV_OVERLAP
    return _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW_SEND_RECV_OVERLAP


def get_ring_ranks_for_intra_window():
    global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW
    if _CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW is None:
        raise AssertionError('Context parallel ranks for ring intra window not initialized')
    return _CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW


def get_ring_ranks_for_inter_window_kv():
    global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV
    if _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV is None:
        raise AssertionError('Context parallel ranks for ring inter window KV not initialized')
    return _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV


def get_ring_ranks_for_inter_window_dkv():
    global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV
    if _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV is None:
        raise AssertionError('Context parallel ranks for ring inter window DKV not initialized')
    return _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV
