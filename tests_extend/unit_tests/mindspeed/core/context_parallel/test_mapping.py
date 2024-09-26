# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import os
import random
import numpy as np
import pytest
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.distributed as dist

from mindspeed import megatron_adaptor
from unit_tests.common import DistributedTest
from mindspeed.core.context_parallel.mapping import (all_to_all,split_forward_gather_backward,
                                                     gather_forward_split_backward)
from megatron.core.parallel_state import destroy_model_parallel, initialize_model_parallel
import megatron.core.parallel_state as mpu
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def cal_split_sizes(dim_size, world_size):
    """
    Calculate the split sizes for a given dimension size and number of processes.

    This function divides the dimension size into `world_size` parts, distributing any remainder
    among the first few parts.

    Args:
        dim_size (int): The total size of the dimension to be split.
        world_size (int): The number of processes (or parts) to split the dimension into.

    Returns:
        List[int]: A list of integers representing the size of each part after splitting.
    """
    split_size = dim_size // world_size
    remainder = dim_size % world_size
    size_list = [split_size + (1 if i < remainder else 0) for i in range(world_size)]
    return size_list


def seed_all(seed=1234, mode=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode)
    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)

seed_all(mode=True)


class TestUnequalSplitGather(DistributedTest):
    world_size = 8

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize("gather_scatter_idx", [(2, 1), (2, 0), (1, 2), (0, 1)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_all_to_all(self, gather_scatter_idx, dtype):
        """
        cases:
        1. gather_idx(2) > scatter_idx(1) > 0
        2. gather_idx(2) > scatter_idx(0) == 0
        3. 0 < gather_idx(1) < scatter_idx(2)
        4. 0 == gather_idx(0) < scatter_idx(1)
        """
        args = parse_args(None, True)
        set_args(args)
        destroy_model_parallel()
        initialize_model_parallel(tensor_model_parallel_size=self.world_size)
        group = mpu.get_tensor_model_parallel_group()
        input_1 = torch.randn(12, 13, 14, 128).cuda().to(dtype)
        input_2 = torch.randn(16, 32, 64, 128).cuda().to(dtype)
        gather_idx, scatter_idx = gather_scatter_idx
        self.run_all_to_all(input_1, scatter_idx, gather_idx, group=group)
        self.run_all_to_all(input_2, scatter_idx, gather_idx, group=group)

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_unequal_split_gather(self, dtype):
        args = parse_args(None, True)
        set_args(args)
        destroy_model_parallel()
        initialize_model_parallel(tensor_model_parallel_size=self.world_size)
        group = mpu.get_tensor_model_parallel_group()
        input_tensor_v1 = torch.randn(16, 8192, 128).cuda().to(dtype)
        input_tensor_v2 = torch.randn(12, 8192, 128).cuda().to(dtype)
        for dim in range(len(input_tensor_v1.size())):
            self.run_split_gather(input_tensor_v1, dim, group)
        for dim in range(len(input_tensor_v2.size())):
            self.run_split_gather(input_tensor_v2, dim, group)

    def run_all_to_all(self, input_, scatter_idx, gather_idx, group):
        world_size = dist.get_world_size(group)
        if (input_.size(scatter_idx) % world_size) != 0:
            s_split_sizes = cal_split_sizes(dim_size=input_.size(scatter_idx),
                                            world_size=world_size)
            t_split_sizes = [input_.size(gather_idx)] * world_size
        else:
            s_split_sizes = None
            t_split_sizes = None
        output_ = all_to_all(input_, group, scatter_idx, gather_idx, s_split_sizes, t_split_sizes)
        output_2 = all_to_all(output_, group, gather_idx, scatter_idx, t_split_sizes, s_split_sizes)
        assert torch.allclose(input_, output_2), "The input and the final output should be close."

    def run_split_gather(self, x, dim, group):
        split_sizes = cal_split_sizes(dim_size=x.size(dim), world_size=dist.get_world_size(group))
        x_split = split_forward_gather_backward(x, group, dim=dim, split_sizes=split_sizes, grad_scale="down")
        x_gather = gather_forward_split_backward(x_split, group, dim=dim, gather_sizes=split_sizes, grad_scale="up")
        tols = dict(atol=5e-3, rtol=5e-3)
        if x.dtype == torch.bfloat16:
            tols = dict(atol=2.5e-2, rtol=2.5e-2)
        assert torch.allclose(x, x_gather, **tols), f"Output:{x_gather} results do not match the input:{x}"
