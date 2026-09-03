# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import math
from typing import List, Optional

import torch

from megatron.core.optimizer.param_layout import PerBufferParamLayout, pad_param_start

from mindspeed.args_utils import get_full_args


def _bucket_end_divisor(params: List[torch.nn.Parameter], data_parallel_world_size: int, ddp_config) -> int:
    """Return an element divisor that preserves Megatron and Ascend alignment contracts."""
    element_sizes = {param.data.element_size() for param in params}
    if len(element_sizes) != 1:
        raise AssertionError("A parameter buffer must contain a single element size.")

    element_size = element_sizes.pop()
    alignment_bytes = get_full_args().param_and_grad_buffer_pad
    alignment_elements = alignment_bytes // math.gcd(alignment_bytes, element_size)

    # Keep every DP shard aligned, matching the legacy feature semantics, and
    # retain Megatron 0.18's baseline 128-element bucket alignment.  The high
    # bandwidth option remains part of the layout contract when enabled.
    divisors = [data_parallel_world_size * alignment_elements, 128]
    if ddp_config.pad_buckets_for_high_nccl_busbw:
        divisors.append(2**16)
    return math.lcm(*divisors)


def compute_aligned_per_buffer_param_layout(
    params: List[torch.nn.Parameter],
    bucket_size: Optional[int],
    data_parallel_world_size: int,
    ddp_config,
    param_indices: Optional[List[int]] = None,
) -> PerBufferParamLayout:
    """Build a Megatron 0.18 parameter layout with configurable bucket alignment."""
    divisor = _bucket_end_divisor(params, data_parallel_world_size, ddp_config)
    param_index_map = {}
    bucket_indices = []
    per_bucket_numel_unpadded = []

    param_start_index = 0
    bucket_start_index = 0
    bucket_params = set()
    bucket_id = 0

    def finalize_bucket(param_end_index):
        nonlocal bucket_start_index, bucket_id
        per_bucket_numel_unpadded.append(param_end_index - bucket_start_index)
        bucket_end_index = ((param_end_index + divisor - 1) // divisor) * divisor
        bucket_indices.append((bucket_start_index, bucket_end_index))
        bucket_start_index = bucket_end_index
        bucket_id += 1
        return bucket_end_index

    for param in params[::-1]:
        param_start_index = pad_param_start(param_start_index)

        requires_own_bucket = getattr(param, "shared_embedding", False)
        if requires_own_bucket and bucket_params:
            param_start_index = finalize_bucket(param_start_index)
            bucket_params = set()

        param_end_index = param_start_index + param.data.nelement()
        param_index_map[param] = (param_start_index, param_end_index, bucket_id)
        bucket_params.add(param)

        if (bucket_size is not None and (param_end_index - bucket_start_index) >= bucket_size) or requires_own_bucket:
            param_start_index = finalize_bucket(param_end_index)
            bucket_params = set()
        else:
            param_start_index = param_end_index

    if bucket_params:
        finalize_bucket(param_end_index)

    return PerBufferParamLayout(
        param_index_map=param_index_map,
        bucket_indices=bucket_indices,
        per_bucket_numel_unpadded=per_bucket_numel_unpadded,
        param_indices=param_indices if param_indices is not None else [],
    )
