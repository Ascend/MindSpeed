# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""MTP packed-token movement without per-document host dispatch."""

from functools import wraps

import torch


def roll_tensor_packed_seq_wrapper(roll_packed_seq):
    """Vectorize Megatron's CP=1 packed roll, retaining its boundary semantics.

    Cumulative boundaries must be ordered and nonnegative, as required by
    PackedSeqParams. Repeated boundaries represent empty documents.
    Keep the prefix before the first boundary and suffix after the last one
    unchanged, just like Megatron's clone-and-slice implementation. In
    particular, do not silently prepend zero to MindSpeed EOD endpoints: that
    would change existing MTP inputs, losses and gradients.
    """

    @wraps(roll_packed_seq)
    def wrapper(tensor, shifts, dims, packed_seq_params, cp_group=None):
        cp_size = cp_group.size() if cp_group is not None else 1
        cu_seqlens = packed_seq_params.cu_seqlens_q
        supported_roll = cp_size == 1 and shifts == -1 and dims in (-1, tensor.dim() - 1)
        if (
            not supported_roll
            or not isinstance(cu_seqlens, torch.Tensor)
            or cu_seqlens.dim() != 1
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
        ):
            return roll_packed_seq(tensor, shifts, dims, packed_seq_params, cp_group)

        # Preserve output layout as well as values so the final sum uses the
        # same layout as the original implementation, including strided inputs.
        rolled_tensor = tensor.clone()
        sequence_length = tensor.shape[-1]
        if cu_seqlens.numel() <= 1 or sequence_length == 0:
            return rolled_tensor, rolled_tensor.sum()

        # Python slices in the original implementation clip endpoints beyond
        # this tensor's last dimension (e.g. batch-global EOD endpoints).
        boundaries = cu_seqlens.to(device=tensor.device, dtype=torch.long, non_blocking=True)
        boundaries = boundaries.clamp(max=sequence_length)
        positions = torch.arange(sequence_length, device=tensor.device)
        processed = (positions >= boundaries[:1]) & (positions < boundaries[-1:])

        # Use exclusive endpoints, then drop slot zero. Empty documents at
        # boundary zero cannot accidentally clear the final token (index -1).
        endpoints = torch.zeros(sequence_length + 1, dtype=torch.bool, device=tensor.device)
        endpoints.index_fill_(0, boundaries[1:], True)
        shifted = torch.roll(tensor, shifts=-1, dims=-1)
        shifted.masked_fill_(endpoints[1:], 0)
        rolled_tensor.copy_(torch.where(processed, shifted, tensor))
        return rolled_tensor, rolled_tensor.sum()

    return wrapper
