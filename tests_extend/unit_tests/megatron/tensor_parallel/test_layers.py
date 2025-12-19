# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
import pytest
import torch
import torch_npu

import mindspeed.megatron_adaptor
import megatron.core.parallel_state as Utils
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.core.tensor_parallel.layers import linear_with_frozen_weight
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

from tests_extend.unit_tests.common import DistributedTest


class TestTPLayers(DistributedTest):
    world_size = 8

    args = parse_args(None, True)
    set_args(args)

    @pytest.mark.parametrize("tensor_parallel, allreduce_dgrad", [(1, False), (8, True)])
    def test_LinearWithFrozenWeight(self, tensor_parallel, allreduce_dgrad):
        Utils.initialize_model_parallel(tensor_parallel, 1)
    
        size_per_partition = int(8 / tensor_parallel)
    
        # Input is an 8x8 identity matrix.
        input_data = torch.eye(8).cuda()
        input_data.requires_grad = True
    
        # Weight is an 8x8 matrix of all ones. If tensor parallelism > 1, the weight is partitioned evenly across GPUs.
        weight = torch.ones((size_per_partition, 8)).cuda()
    
        # Bias is a vector of length 8 of all zeros. If tensor parallelism > 1, the bias is partitioned evenly across GPUs
        bias = torch.zeros((size_per_partition)).cuda()
    
        gradient_accumulation_fusion = False
        async_grad_allreduce = allreduce_dgrad
        sequence_parallel = False
        grad_output_buffer = None
        wgrad_deferral_limit = None
        tp_group = None

        output_parallel = linear_with_frozen_weight(
            input_data,
            weight,
            bias,
            gradient_accumulation_fusion,
            async_grad_allreduce,
            sequence_parallel,
            tp_group,
            grad_output_buffer,
            wgrad_deferral_limit,
            allreduce_dgrad,
        )
        output = gather_from_tensor_model_parallel_region(
            output_parallel
        )  # no-op if tensor_parallel == 1.
        output.sum().backward()
    
        expected_output = torch.ones(8).cuda()
        expected_grad = 8 * torch.ones(8).cuda()
    
        assert torch.allclose(output, expected_output)
        assert torch.allclose(input_data.grad, expected_grad)
    
        Utils.destroy_model_parallel()
