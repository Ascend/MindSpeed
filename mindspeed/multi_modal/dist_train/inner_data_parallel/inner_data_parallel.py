# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
from .. import parallel_state as dist_ps


@dist_ps.subwrold_decorator
def get_inner_data_parallel_group():
    """Get the inner data parallel group the caller rank belongs to."""
    if dist_ps._INNER_DATA_PARALLEL_GROUP is None:
        raise RuntimeError('inner data parallel group is not initialized')
    return dist_ps._INNER_DATA_PARALLEL_GROUP


@dist_ps.subwrold_decorator
def get_inner_data_parallel_world_size():
    """Return world size for the inner data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(
            group=get_inner_data_parallel_group()
        )
    else:
        return 0


@dist_ps.subwrold_decorator
def get_inner_data_parallel_rank():
    """Return my rank for the inner data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(
            group=get_inner_data_parallel_group()
        )
    else:
        return 0


def get_inner_data_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank in the inner data parallel group."""
    if dist_ps._CUR_SUB_WORLD is None:
        return 0
    global_rank = (torch.distributed.get_rank() - dist_ps._CUR_SUB_WORLD.start_rank)
    local_world_size = get_inner_data_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size + dist_ps._CUR_SUB_WORLD.start_rank
