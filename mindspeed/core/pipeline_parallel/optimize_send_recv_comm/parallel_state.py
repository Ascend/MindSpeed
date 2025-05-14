"""Parallel state handling for send/recv communication optimization.

Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""

from typing import Callable, Optional, List

import torch
import yaml

_PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM = None


def initialize_model_parallel_impl(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    hierarchical_context_parallel_sizes: Optional[List[int]] = None,
    expert_model_parallel_size: int = 1,
    num_distributed_optimizer_instances: int = 1,
    expert_tensor_parallel_size: Optional[int] = None,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-cp-ep-dp-pp",
    encoder_tensor_model_parallel_size: Optional[int] = 0,
    encoder_pipeline_model_parallel_size: Optional[int] = 0,
    get_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
    get_position_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
    get_nccl_options: Optional[Callable] = None,
):
    """Initialize pipeline model parallel group."""
    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)
    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM
    if _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM is not None:
        raise AttributeError(
            "Pipeline parallel group for new stream is already initialized"
        )
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    if get_nccl_options is not None:
        for i in range(num_pipeline_model_parallel_groups):
            ranks = range(i, world_size, num_pipeline_model_parallel_groups)
            group = torch.distributed.new_group(
                ranks,
                pg_options=get_nccl_options(
                    "pp_new_stream",
                    nccl_comm_cfgs,
                ),
            )
            if rank in ranks:
                _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM = group


def get_pipeline_parallel_group_for_new_stream():
    """Get pipeline model parallel group for new stream."""
    if _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM is None:
        raise AttributeError("Pipeline parallel group of backward is not initialized")
    return _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM
