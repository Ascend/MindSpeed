# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from mindspeed.core.tensor_parallel.comm_utils import (
    _split_along_first_dim,
    sync_gather_along_first_dim,
    sync_reduce_scatter_along_first_dim
)
from mindspeed.core.tensor_parallel.comm_group_api import CollectiveCommIntf
from mindspeed.core.multi_modal.dist_train import dist_parallel_state as dist_ps


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



def gather_from_inner_dp_region(input_, inner_dp_parallel_output_grad=True):
    return _GatherFromInnerDataParallelRegion.apply(input_, inner_dp_parallel_output_grad)


class _GatherFromInnerDataParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, inner_dp_parallel_output_grad=True):
        return sync_gather_along_first_dim(input_, InnerDPCollectiveComm)

    @staticmethod
    def forward(ctx, input_, inner_dp_parallel_output_grad=True):
        ctx.inner_dp_parallel_output_grad = inner_dp_parallel_output_grad
        return sync_gather_along_first_dim(input_, InnerDPCollectiveComm)

    @staticmethod
    def backward(ctx, grad_output):
        inner_dp_parallel_output_grad = ctx.inner_dp_parallel_output_grad

        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce
        # scattered and whereas if the computation is duplicated,
        # output gradients need to be scattered.
        if inner_dp_parallel_output_grad:
            return sync_reduce_scatter_along_first_dim(grad_output, InnerDPCollectiveComm), None
        else:
            return _split_along_first_dim(grad_output, InnerDPCollectiveComm), None


class InnerDPCollectiveComm(CollectiveCommIntf):
    def __init__(self, name='inner-dp'):
        super().__init__(name)

    @classmethod
    def get_comm_rank(cls):
        return get_inner_data_parallel_rank()

    @classmethod
    def get_comm_group_world_size(cls):
        return get_inner_data_parallel_world_size()

    @classmethod
    def get_comm_group(cls):
        return get_inner_data_parallel_group()


def split_data(data: torch.Tensor, padding_val: int = 0):
    data_num = data.shape[0]
    dp_size = get_inner_data_parallel_world_size()
    data_pad_num = (data_num + dp_size - 1) // dp_size * dp_size
    data_shape = tuple(data.shape[1:])

    # Flatten the data associated with the keys
    flatten_data = data.contiguous().view(-1, *data_shape).cuda()

    if padding_val != 0:
        flatten_data_pad = torch.full((data_pad_num, *data_shape), padding_val, device=torch.cuda.current_device())
    else:
        flatten_data_pad = torch.zeros((data_pad_num, *data_shape), device=torch.cuda.current_device())

    flatten_data_pad[:data_num, ...] = flatten_data
    split_data_list = list(torch.chunk(flatten_data_pad, dp_size, dim=0))
    data = split_data_list[get_inner_data_parallel_rank()]

    return data
