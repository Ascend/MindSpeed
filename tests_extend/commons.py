# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import os
import random
import torch
import torch_npu
import numpy

import megatron.core.parallel_state as ps


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)


def set_deterministic(seed):
    """ set npu deterministic mode
    'HCCL_DETERMINISTIC' is a deterministic switch in ops level, set it to 'True' to enable ops level deterministic, set it to 'False' to disable ops level deterministic.
    'CLOSE_MATMUL_K_SHIFT' is a switch of matmul K-axis shift, set it to '1' to close matmul K-axis shift, set it to '0' to enable matmul K-axis shift.
    'PYTHONHASHSEED' refers to python hash seed, use a string of non-negative integer to specify the seed.
    """
    os.environ['HCCL_DETERMINISTIC'] = 'True'
    os.environ['CLOSE_MATMUL_K_SHIFT'] = '1'
    torch.use_deterministic_algorithms(True)
    set_random_seed(seed)


def initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    pipeline_model_parallel_split_rank=None,
    context_parallel_size=1,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None
):
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank=pipeline_model_parallel_split_rank,
        context_parallel_size=context_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        expert_tensor_parallel_size=expert_tensor_parallel_size
    )
