# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from functools import lru_cache
from typing import Optional

import torch
from torch.distributed import ProcessGroup as dist_group_type
from einops import rearrange


@lru_cache
def get_distributed_world_size(group: Optional[dist_group_type] = None) -> int:
    """Return world size for the distributed group."""
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=group)


@lru_cache
def get_distributed_rank(group: Optional[dist_group_type] = None) -> int:
    """Return my rank for the distributed group."""
    assert torch.distributed.is_initialized(), "torch.distributed is not initialized."
    return torch.distributed.get_rank(group=group)


def prepare_sbhd_format(query_layer, key_layer, value_layer):
    """Prepare tensors for SBHD format"""
    _, _, n_head, _ = query_layer.shape

    query_layer, key_layer, value_layer = [
        rearrange(x, 's b h d -> s b (h d)')
        for x in [query_layer, key_layer, value_layer]
    ]

    return query_layer, key_layer, value_layer, n_head


def prepare_thd_format(query_layer, cu_seqlens_q, cu_seqlens_kv):
    """Prepare tensors for THD format"""
    _, n_head, _ = query_layer.shape

    # Convert to list if tensor
    if isinstance(cu_seqlens_q, torch.Tensor):
        cu_seqlens_q = cu_seqlens_q.tolist()
    if isinstance(cu_seqlens_kv, torch.Tensor):
        cu_seqlens_kv = cu_seqlens_kv.tolist()

    return n_head, cu_seqlens_q, cu_seqlens_kv