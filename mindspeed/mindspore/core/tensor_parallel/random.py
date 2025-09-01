# Copyright (c) 2022; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
from torch.autograd import recompute_instance
from torch.utils.checkpoint import detach_variable
from megatron.core.tensor_parallel import get_cuda_rng_tracker

from megatron.core.utils import safely_set_viewless_tensor_data
from megatron.core.tensor_parallel import gather_split_1d_tensor, split_tensor_into_1d_equal_chunks


def local_set_cuda_rng_state(new_state, device=-1, graph_safe: bool = False): # -_set_cuda_rng_state , 有个问题，这个同时是helper function不知道会不会有问题
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    torch.cuda.set_rng_state(new_state)

