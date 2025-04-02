# Copyright (c) 2022; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_src_rank,
)

from megatron.core.tensor_parallel.data import _MAX_DATA_DIM


def local_build_key_size_numel_dictionaries(keys, data):
    """Build the size on rank 0 and broadcast."""
    max_dim = _MAX_DATA_DIM
    sizes = [0] * (max_dim * len(keys))

    # Pack the sizes on rank zero.
    if get_tensor_model_parallel_rank() == 0:
        offset = 0
        for key in keys:
            assert data[key].dim() < max_dim, 'you should increase MAX_DATA_DIM'
            size = data[key].size()
            for i, s in enumerate(size):
                sizes[i + offset] = s
            offset += max_dim

    # Move to GPU and broadcast.
    sizes_cuda = torch.tensor(sizes, dtype=torch.long, device='cuda')
    torch.distributed.broadcast(
        sizes_cuda, get_tensor_model_parallel_src_rank(), group=get_tensor_model_parallel_group()
    )

    # Move back to cpu and unpack.
    sizes_np = sizes_cuda.numpy()
    key_size = {}
    key_numel = {}
    total_numel = 0
    offset = 0
    for key in keys:
        i = 0
        size = []
        numel = 1
        while sizes_np[offset + i] > 0:
            this_size = sizes_np[offset + i]
            size.append(this_size)
            numel *= this_size
            i += 1
        key_size[key] = size
        key_numel[key] = numel
        total_numel += numel
        offset += max_dim

    return key_size, key_numel, total_numel
