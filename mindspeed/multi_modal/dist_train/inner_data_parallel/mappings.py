# Copied from Megatron-LM: https://github.com/NVIDIA/Megatron-LM
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
from mindspeed.core.tensor_parallel.comm_utils import (
    _split_along_first_dim,
    sync_gather_along_first_dim,
    sync_reduce_scatter_along_first_dim
)
from mindspeed.core.tensor_parallel.comm_group_api import CollectiveCommIntf
from .inner_data_parallel import (
    get_inner_data_parallel_group,
    get_inner_data_parallel_world_size,
    get_inner_data_parallel_rank,
)


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
