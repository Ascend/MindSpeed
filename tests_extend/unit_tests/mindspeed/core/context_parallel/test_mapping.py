# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import os
from typing import Optional, List
import time
import random
from functools import reduce
import operator

import numpy as np
import pytest
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.distributed as dist

from mindspeed import megatron_adaptor
from tests_extend.unit_tests.common import DistributedTest
from mindspeed.core.context_parallel.ulysses_context_parallel.unaligned_cp.mapping import (all_to_all, split_forward_gather_backward,
                                                                                   gather_forward_split_backward, cal_split_sizes)
from megatron.core.parallel_state import destroy_model_parallel, initialize_model_parallel
import megatron.core.parallel_state as mpu
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


def seed_all(seed=1234, mode=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode)
    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)


seed_all(mode=True)


def _all_to_all_standard(
        input_: torch.Tensor,
        group: dist.ProcessGroup,
        scatter_dim: int,
        gather_dim: int,
        scatter_sizes: Optional[List[int]] = None,
        gather_sizes: Optional[List[int]] = None
):
    """
    Helper function to perform the all-to-all operation. It scatters the input tensor along the specified scatter
    dimension and then gathers it along the specified gather dimension. This function supports non-uniform scatter
    and gather sizes.

    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        group (dist.ProcessGroup): The process group to perform the operation within.
        scatter_dim (int): The index of the dimension that needs to be scattered.
        gather_dim (int): The index of the dimension that needs to be gathered.
        scatter_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be scattered.
            If not provided, the tensor will be split equally among the processes. Defaults to None.
        gather_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be gathered.
            If not provided, the tensor will be assumed to have equal parts. Defaults to None.

    Returns:
        torch.Tensor: The resulting tensor after performing the all-to-all operation.
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    # Ensure scatter_sizes and gather_sizes are lists if provided
    assert scatter_sizes is None or isinstance(scatter_sizes, list)
    assert gather_sizes is None or isinstance(gather_sizes, list)

    # Split the input tensor based on scatter_sizes or equally if scatter_sizes is None
    if scatter_sizes:
        input_list = [t.contiguous() for t in torch.split(input_, scatter_sizes, scatter_dim)]
    else:
        input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]

    # Prepare the output list with appropriate shapes
    if gather_sizes:
        output_list = []
        tensor_shape_base = input_list[rank].size()
        for i in range(world_size):
            tensor_shape = list(tensor_shape_base)
            tensor_shape[gather_dim] = gather_sizes[i]
            output_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))

    else:
        output_list = [torch.empty_like(input_list[0], dtype=input_.dtype, device=input_.device)
                       for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)

    output = torch.cat(output_list, dim=gather_dim).contiguous()
    return output


def measure_time(func, *args, **kwargs):
    start_time = time.time()
    res = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, res


class TestUnevenSplitGather(DistributedTest):
    world_size = 8
    warm_up_time = 1
    total_time = 3

    def test_setup(self):
        assert int(os.environ["WORLD_SIZE"]) == self.world_size
        args = parse_args(None, True)
        set_args(args)
        destroy_model_parallel()
        initialize_model_parallel(tensor_model_parallel_size=self.world_size)

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize("gather_scatter_idx", [(3, 0), (3, 1), (3, 2),
                                                    (2, 0), (2, 1), (2, 3),
                                                    (1, 0), (1, 2), (1, 3),
                                                    (0, 1), (0, 2), (0, 3)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_all_to_all(self, gather_scatter_idx, dtype):
        args = parse_args(None, True)
        set_args(args)
        destroy_model_parallel()
        initialize_model_parallel(tensor_model_parallel_size=self.world_size)
        input_shapes = [[32, 64, 32], [32, 64, 32, 32],
                        [29, 51, 57], [29, 51, 65, 57]]
        group = mpu.get_tensor_model_parallel_group()
        gather_idx, scatter_idx = gather_scatter_idx
        for input_shape in input_shapes:
            input_ = torch.randn(input_shape).cuda().to(dtype)
            num_dims = input_.dim()
            if gather_idx >= num_dims or scatter_idx >= num_dims:
                return
            scatter_size_list = None
            gather_sizes_list = None
            if (input_.size(scatter_idx) % self.world_size) != 0:
                scatter_size_list = cal_split_sizes(dim_size=input_.size(scatter_idx), world_size=self.world_size)
                gather_sizes_list = [input_.size(gather_idx)] * self.world_size
            self.run_all_to_all(
                input_, scatter_idx, gather_idx, group, scatter_size_list, gather_sizes_list)

    @pytest.mark.parametrize("gather_scatter_idx", [(3, 0), (3, 1), (3, 2),
                                                    (2, 0), (2, 1), (2, 3),
                                                    (1, 0), (1, 2), (1, 3),
                                                    (0, 1), (0, 2), (0, 3)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_all_to_all_full_unaligned(self, gather_scatter_idx, dtype):
        args = parse_args(None, True)
        set_args(args)
        destroy_model_parallel()
        initialize_model_parallel(tensor_model_parallel_size=self.world_size)
        group = mpu.get_tensor_model_parallel_group()
        rank = dist.get_rank(group)
        unscatter_gathered_shapes = [[9, 21, 15], [9, 21, 15, 10]]
        group = mpu.get_tensor_model_parallel_group()
        gather_idx, scatter_idx = gather_scatter_idx
        for unscatter_gathered_shape in unscatter_gathered_shapes:
            num_dims = len(unscatter_gathered_shape)
            if gather_idx >= num_dims or scatter_idx >= num_dims:
                return
            gather_size = unscatter_gathered_shape[gather_idx]
            scatter_size = unscatter_gathered_shape[scatter_idx]
            gather_size_list = cal_split_sizes(dim_size=gather_size, world_size=self.world_size)
            scatter_size_list = cal_split_sizes(dim_size=scatter_size, world_size=self.world_size)
            input_shape = unscatter_gathered_shape
            input_shape[gather_idx] = gather_size_list[rank]
            total_elements = reduce(operator.mul, input_shape, 1)
            input_ = normalize_tensor(torch.arange(total_elements).reshape(input_shape)).cuda().to(dtype) + rank

            expected_output_list = []
            for mock_rank in range(self.world_size):
                input_shape[gather_idx] = gather_size_list[mock_rank]
                total_elements = reduce(operator.mul, input_shape, 1)
                mock_input_tensor = normalize_tensor(torch.arange(total_elements).reshape(input_shape)).cuda().to(dtype) + mock_rank
                if mock_rank == rank:
                    assert torch.equal(mock_input_tensor, input_)
                tensor_list = torch.split(mock_input_tensor, scatter_size_list, dim=scatter_idx)
                expected_output_list.append(tensor_list[rank].contiguous())
            expected = torch.cat(expected_output_list, dim=gather_idx).contiguous()
            result = all_to_all(input_, group, scatter_dim=scatter_idx, gather_dim=gather_idx, gather_size=gather_size)
            assert torch.equal(result, expected)

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize("input_shape", [[32, 64, 32], [32, 64, 32, 32],
                                             [29, 51, 57], [29, 51, 65, 57]])
    @pytest.mark.parametrize("dim", [0, 1, 2, 3])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_split_gather_default(self, input_shape, dim, dtype):
        args = parse_args(None, True)
        set_args(args)
        destroy_model_parallel()
        initialize_model_parallel(tensor_model_parallel_size=self.world_size)
        group = mpu.get_tensor_model_parallel_group()
        input_tensor = torch.randn(input_shape).cuda().to(dtype)
        if dim >= input_tensor.dim():
            return
        self.run_split_gather(input_tensor, dim, group)

    def run_all_to_all(self, input_, scatter_idx, gather_idx, group, scatter_size_list, gather_sizes_list):
        input_clone = input_.clone()
        for i in range(TestUnevenSplitGather.total_time):
            all_to_all_time1, standard_output1 = measure_time(
                _all_to_all_standard, input_, group, scatter_idx, gather_idx, scatter_size_list, gather_sizes_list)
            all_to_all_time2, standard_output2 = measure_time(
                _all_to_all_standard, standard_output1, group, gather_idx, scatter_idx,
                gather_sizes_list, scatter_size_list)
            assert torch.equal(input_, standard_output2), "The input_ and standard_output2 should be close."
            all_to_all_single_time1, output1 = measure_time(
                all_to_all, input_clone, group, scatter_idx, gather_idx)
            all_to_all_single_time2, output2 = measure_time(
                all_to_all, output1, group, gather_idx, scatter_idx, gather_size=input_clone.size(scatter_idx)
                if input_clone.size(scatter_idx) % self.world_size != 0 else None)
            assert torch.equal(input_clone, output2), "The input_clone and output2 should be close."
            if i < TestUnevenSplitGather.warm_up_time:
                continue
            assert torch.equal(standard_output1, output1), "The standard_output1 and output1 should be close."
            assert torch.equal(standard_output2, output2), "The standard_output2 and output2 should be close."

    def run_split_gather(self, x, dim, group):
        split_sizes = None
        world_size = dist.get_world_size(group)
        if (x.size(dim) % world_size) != 0:
            split_sizes = cal_split_sizes(dim_size=x.size(dim), world_size=dist.get_world_size(group))
        x_split = split_forward_gather_backward(x, group, dim=dim, split_sizes=split_sizes, grad_scale="down")
        x_gather = gather_forward_split_backward(x_split, group, dim=dim, gather_sizes=split_sizes, grad_scale="up")
        assert torch.equal(x, x_gather), f"Output:{x_gather} results do not match the input:{x}"
