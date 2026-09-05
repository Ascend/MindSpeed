# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import torch
import torch.distributed as dist


def _validate_sequence_tensor_and_positive_length(local_tensor: torch.Tensor, positive_length: int):
    if not torch.is_tensor(local_tensor):
        raise TypeError("local_tensor must be a torch.Tensor.")
    if local_tensor.dim() == 0:
        raise ValueError("local_tensor must have a sequence dimension.")
    if positive_length <= 0:
        raise ValueError("positive_length must be a positive integer.")


def _get_cp_size_and_rank(cp_group):
    if not dist.is_available() or not dist.is_initialized():
        return 1, 0
    if cp_group is None:
        return dist.get_world_size(), dist.get_rank()
    return dist.get_world_size(group=cp_group), dist.get_rank(group=cp_group)


def _get_total_seq_len(local_seq_len, cp_size, cu_seqlens):
    if cu_seqlens is None:
        return local_seq_len * cp_size
    if torch.is_tensor(cu_seqlens):
        if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must contain at least a start and end offset.")
        return int(cu_seqlens[-1].item())
    cu_seqlens = _to_int_list(cu_seqlens)
    if len(cu_seqlens) < 2:
        raise ValueError("cu_seqlens must contain at least a start and end offset.")
    return cu_seqlens[-1]


def _to_int_list(values):
    if torch.is_tensor(values):
        values = values.detach().cpu().tolist()
    return [int(value) for value in values]


def _to_optional_tuple(values):
    if values is None:
        return None
    return tuple(_to_int_list(values))


def _to_int32_tensor(values, device):
    if torch.is_tensor(values):
        return values.to(device=device, dtype=torch.int32)
    return torch.tensor(_to_int_list(values), dtype=torch.int32, device=device)


def normalize_cu_seqlens(
    cu_seqlens,
    device,
    *,
    trusted=False,
    name="cu_seqlens",
    require_leading_zero=False,
):
    """Normalize cu_seqlens into a validated on-device int32 tensor.

    Accepts ``None`` (returned unchanged), Python sequences, and tensors.  When
    ``trusted`` is set and the input is already a tensor, the int32/device
    conversion is applied but shape/ordering validation is skipped.  Non-tensor
    inputs are always materialized and validated because they cannot be trusted
    before conversion.

    Packed global offsets must describe the flattened sequence starting at 0.
    ``require_leading_zero=True`` raises instead of prepending 0, so a shard
    that starts at a non-zero global offset cannot be silently rewritten into
    an extra sample.
    """
    if cu_seqlens is None:
        return None
    if not torch.is_tensor(cu_seqlens):
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        trusted = False
    else:
        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32)
    if trusted:
        return cu_seqlens
    if cu_seqlens.dim() != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if cu_seqlens.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if require_leading_zero and cu_seqlens.numel() < 2:
        raise ValueError(f"{name} must contain at least a start and end offset.")
    if int(cu_seqlens[0].item()) != 0:
        if require_leading_zero:
            raise ValueError(f"{name} must start with 0.")
        cu_seqlens = torch.cat((cu_seqlens.new_zeros(1), cu_seqlens))
    if cu_seqlens.numel() > 1 and torch.any(torch.diff(cu_seqlens) < 0):
        raise ValueError(f"{name} must be monotonically non-decreasing.")
    return cu_seqlens


def _lengths_from_cu_seqlens(cu_seqlens):
    if cu_seqlens.dim() != 1:
        raise ValueError("cu_seqlens must be a 1-D tensor.")
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least a start and end offset.")
    return torch.diff(cu_seqlens).to(dtype=torch.int32)


def enumerate_global_block_starts(sample_boundaries, compression_ratio):
    """Enumerate global compressed-block start positions from sample boundaries.

    ``sample_boundaries`` is an int64 cu_seqlens-like tensor (one boundary per
    sample). Returns a 1-D int64 tensor of every compressed-block start position
    on the same device, ordered by sample and then by block.
    """
    sample_starts = sample_boundaries[:-1]
    block_counts = torch.div(
        torch.diff(sample_boundaries),
        int(compression_ratio),
        rounding_mode="floor",
    )
    candidate_count = int(block_counts.sum())
    if candidate_count == 0:
        return torch.empty(0, dtype=torch.long, device=sample_boundaries.device)

    sample_ids = torch.repeat_interleave(
        torch.arange(sample_starts.numel(), dtype=torch.long, device=sample_boundaries.device),
        block_counts,
        output_size=candidate_count,
    )
    sample_candidate_offsets = torch.repeat_interleave(
        torch.cumsum(block_counts, dim=0) - block_counts,
        block_counts,
        output_size=candidate_count,
    )
    block_offsets = (
        torch.arange(candidate_count, dtype=torch.long, device=sample_boundaries.device) - sample_candidate_offsets
    )
    return sample_starts.index_select(0, sample_ids) + block_offsets * int(compression_ratio)


def _build_deepseek_v4_cmp_visibility(
    query_positions,
    block_starts,
    valid_mask,
    compression_ratio,
    cu_seqlens,
):
    query_positions = query_positions.to(dtype=torch.long)
    block_starts = block_starts.to(device=query_positions.device, dtype=torch.long)
    valid_mask = valid_mask.to(device=query_positions.device, dtype=torch.bool)
    if block_starts.shape != valid_mask.shape:
        raise ValueError(
            "block_starts and valid_mask must have the same shape: "
            f"got {tuple(block_starts.shape)} and {tuple(valid_mask.shape)}"
        )

    query_count = query_positions.numel()
    block_count = block_starts.numel()
    if query_count == 0 or block_count == 0:
        return torch.zeros(
            (query_count, block_count),
            dtype=torch.bool,
            device=query_positions.device,
        )

    if cu_seqlens is None:
        sample_boundaries = _get_sample_boundaries_tensor(
            None,
            _infer_total_seq_len(query_positions, block_starts, compression_ratio),
            query_positions.device,
        )
    else:
        sample_boundaries = _get_sample_boundaries_tensor(
            cu_seqlens,
            0,
            query_positions.device,
        )
    valid_blocks = valid_mask & (block_starts >= 0)
    block_ends = block_starts + compression_ratio
    query_sample_idx = torch.searchsorted(sample_boundaries, query_positions, right=True) - 1
    block_sample_idx = torch.searchsorted(sample_boundaries, block_starts, right=True) - 1
    block_end_sample_idx = torch.searchsorted(sample_boundaries, block_ends - 1, right=True) - 1

    query_in_boundary = (query_sample_idx >= 0) & (query_sample_idx < sample_boundaries.numel() - 1)
    block_in_boundary = (
        valid_blocks
        & (block_sample_idx >= 0)
        & (block_sample_idx < sample_boundaries.numel() - 1)
        & (block_sample_idx == block_end_sample_idx)
    )
    return (
        query_in_boundary.view(-1, 1)
        & block_in_boundary.view(1, -1)
        & (query_sample_idx.view(-1, 1) == block_sample_idx.view(1, -1))
        & (block_ends.view(1, -1) <= (query_positions + 1).view(-1, 1))
    )


def _reshape_deepseek_v4_sparse_indices_for_layout(flat_sparse_indices, q, layout_q):
    topk = flat_sparse_indices.shape[-1]
    if layout_q == "BSND":
        batch_size, seq_len = int(q.shape[0]), int(q.shape[1])
        return flat_sparse_indices.reshape(batch_size, seq_len, 1, topk)
    return flat_sparse_indices.unsqueeze(1)


def _get_sample_boundaries_tensor(cu_seqlens, total_seq_len, device):
    if cu_seqlens is None:
        return torch.tensor([0, total_seq_len], dtype=torch.long, device=device)
    return _to_int32_tensor(cu_seqlens, device).to(dtype=torch.long)


def _infer_total_seq_len(query_positions, block_starts, compression_ratio):
    ends = []
    if query_positions is not None and query_positions.numel() > 0:
        ends.append(int(query_positions.max().item()) + 1)
    if block_starts is not None and block_starts.numel() > 0:
        valid_starts = block_starts.to(dtype=torch.long)
        valid_starts = valid_starts[valid_starts >= 0]
        if valid_starts.numel() > 0:
            ends.append(int(valid_starts.max().item()) + compression_ratio)
    return max(ends) if ends else 0
