# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# pylint: disable=too-many-lines

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch

from ._types import (
    DeepSeekV4CPCompressContext,
    DeepSeekV4CPCompressedKV,
    DeepSeekV4CPCompressionCandidates,
    DeepSeekV4CPMetadata,
    DeepSeekV4CPPendingCompressedKV,
)
from ._utils import (
    _get_cp_size_and_rank,
    _get_sample_boundaries_tensor,
    _get_total_seq_len,
    _to_int_list,
    _to_optional_tuple,
    _validate_sequence_tensor_and_positive_length,
    enumerate_global_block_starts,
)
from ._distributed import (
    _AllGatherCompressedKVAsync,
    _AsyncCollectiveState,
    _ReceivePreviousTail,
    _resolve_previous_tail,
    _resolve_global_ranks,
)


@dataclass(frozen=True)
class _CandidatePlan:
    """CPU-built padded metadata consumed by the device-side batched gathers."""

    candidate_starts: torch.Tensor
    candidate_offsets: torch.Tensor
    valid_mask: torch.Tensor
    candidate_count: int
    candidate_positions: torch.Tensor
    candidate_position_max: int
    candidate_sample_positions: torch.Tensor
    candidate_sample_position_max: int
    sample_boundaries: torch.Tensor
    left_context_starts: Optional[torch.Tensor] = None
    left_context_offsets: Optional[torch.Tensor] = None
    left_context_valid_mask: Optional[torch.Tensor] = None
    sample_ids: Optional[torch.Tensor] = None
    left_context_source_indices: Optional[torch.Tensor] = None
    left_context_boundary_mask: Optional[torch.Tensor] = None
    left_context_boundary_indices: Optional[torch.Tensor] = None
    left_context_reuse_segments: Optional[Sequence[tuple]] = None


@dataclass(frozen=True)
class _DeterministicSelectionPlan:
    """Cached global selection tensors for a contiguous CP partition."""

    selected_indices: torch.Tensor
    valid_count: int
    valid_mask: torch.Tensor
    block_starts: torch.Tensor
    source_rank: torch.Tensor
    rank_candidate_starts: Sequence[torch.Tensor]
    is_identity_compact_order: bool


def prepare_deepseek_v4_compression_candidates_for_cp(
    local_kv: torch.Tensor,
    compression_ratio: int,
    cp_group=None,
    cp_global_ranks: Optional[Sequence[int]] = None,
    seq_dim: int = 0,
    cu_seqlens: Optional[Iterable[int]] = None,
    local_seq_offset: Optional[int] = None,
    output_size: Optional[int] = None,
    include_left_context: bool = False,
    previous_tail: Optional[torch.Tensor] = None,
    use_compact_candidate_view: bool = False,
    batch_shared_sequence: bool = False,
) -> DeepSeekV4CPCompressionCandidates:
    """Prepare reusable candidate blocks and global selection metadata once."""
    _validate_sequence_tensor_and_positive_length(local_kv, compression_ratio)
    normalized_seq_dim = int(seq_dim) % local_kv.dim()
    if batch_shared_sequence and normalized_seq_dim != 1:
        raise ValueError("batch-shared BSND compression requires seq_dim=1 with batch on dimension 0.")

    local_kv_seq_first_view = torch.movedim(local_kv, seq_dim, 0)
    seq_first_input_is_contiguous = local_kv_seq_first_view.is_contiguous()
    local_kv_seq_first = local_kv_seq_first_view.contiguous()
    local_seq_len = local_kv_seq_first.shape[0]
    cp_size, cp_rank = _get_cp_size_and_rank(cp_group)
    if local_seq_offset is None:
        local_seq_offset = cp_rank * local_seq_len

    candidate_cu_seqlens = cu_seqlens
    if batch_shared_sequence:
        total_seq_len, candidate_cu_seqlens = _resolve_batch_shared_sequence_cu_seqlens(
            local_seq_len,
            cp_size,
            cu_seqlens,
            int(local_kv.shape[0]),
        )
    else:
        total_seq_len = _get_total_seq_len(local_seq_len, cp_size, cu_seqlens)
    if output_size is None:
        output_size = total_seq_len // compression_ratio

    previous_context_len = compression_ratio * 2 if include_left_context else compression_ratio
    prev_tail = _resolve_previous_tail(
        local_kv_seq_first,
        previous_context_len,
        previous_tail,
        seq_dim,
    )
    if prev_tail is None:
        prev_tail = _ReceivePreviousTail.apply(
            local_kv_seq_first,
            previous_context_len,
            cp_group,
            _resolve_global_ranks(cp_group, cp_size, cp_global_ranks),
        )
    extended_start = local_seq_offset - previous_context_len

    candidate_capacity = local_seq_len // compression_ratio + 1
    candidate_plan = _build_local_candidate_plan(
        local_seq_offset,
        local_seq_len,
        compression_ratio,
        candidate_cu_seqlens,
        total_seq_len,
        candidate_capacity,
        extended_start,
        prev_tail.shape[0] + local_seq_len,
        include_left_context,
    )
    local_block_starts = candidate_plan.candidate_starts.to(device=local_kv.device)
    local_valid_mask = candidate_plan.valid_mask.to(device=local_kv.device)
    compact_candidate_input = _can_use_compact_candidate_view(
        use_compact_candidate_view,
        seq_first_input_is_contiguous,
        local_seq_offset,
        local_seq_len,
        compression_ratio,
        total_seq_len,
        candidate_plan,
        prev_tail.shape[0],
        include_left_context,
    )
    context_candidate_count = candidate_plan.candidate_count if compact_candidate_input else candidate_capacity
    context_slice = slice(0, context_candidate_count)
    context_candidate_starts = local_block_starts[context_slice]
    context_valid_mask = local_valid_mask[context_slice]
    candidate_positions = candidate_plan.candidate_positions[context_slice].to(device=local_kv.device)
    candidate_sample_positions = candidate_plan.candidate_sample_positions[context_slice].to(device=local_kv.device)

    extended_kv = None
    if compact_candidate_input:
        candidate_blocks = local_kv_seq_first.view(
            context_candidate_count,
            compression_ratio,
            *local_kv_seq_first.shape[1:],
        )
    else:
        extended_kv = torch.cat((prev_tail, local_kv_seq_first), dim=0)
        candidate_blocks = _gather_padded_blocks(
            extended_kv,
            candidate_plan.candidate_offsets.to(device=local_kv.device),
            local_valid_mask,
            compression_ratio,
        )

    left_context_blocks = None
    left_context_boundary_blocks = None
    left_context_starts = None
    left_context_valid_mask = None
    sample_ids = None
    left_context_source_indices = None
    left_context_boundary_mask = None
    left_context_boundary_indices = None
    left_context_reuse_segments = None
    if include_left_context:
        left_context_starts = candidate_plan.left_context_starts[context_slice].to(device=local_kv.device)
        left_context_valid_mask = candidate_plan.left_context_valid_mask[context_slice].to(device=local_kv.device)
        sample_ids = candidate_plan.sample_ids[context_slice].to(device=local_kv.device)
        left_context_source_indices = candidate_plan.left_context_source_indices[context_slice].to(
            device=local_kv.device
        )
        left_context_boundary_mask = candidate_plan.left_context_boundary_mask[context_slice].to(device=local_kv.device)
        left_context_boundary_indices = candidate_plan.left_context_boundary_indices.to(device=local_kv.device)
        left_context_reuse_segments = candidate_plan.left_context_reuse_segments
        if compact_candidate_input:
            left_context_boundary_blocks = _view_compact_left_context_boundaries(
                prev_tail,
                candidate_plan,
                compression_ratio,
            )
        else:
            left_context_blocks = _gather_padded_blocks(
                extended_kv,
                candidate_plan.left_context_offsets.to(device=local_kv.device),
                left_context_valid_mask,
                compression_ratio,
            )

    compress_context = DeepSeekV4CPCompressContext(
        candidate_starts=context_candidate_starts,
        valid_mask=context_valid_mask,
        local_seq_offset=int(local_seq_offset),
        local_seq_len=int(local_seq_len),
        total_seq_len=int(total_seq_len),
        compression_ratio=int(compression_ratio),
        candidate_capacity=int(candidate_capacity),
        cp_size=int(cp_size),
        cp_rank=int(cp_rank),
        seq_dim=int(seq_dim),
        cu_seqlens=_to_optional_tuple(candidate_cu_seqlens),
        left_context_blocks=left_context_blocks,
        left_context_boundary_blocks=left_context_boundary_blocks,
        left_context_starts=left_context_starts,
        left_context_valid_mask=left_context_valid_mask,
        sample_ids=sample_ids,
        left_context_source_indices=left_context_source_indices,
        left_context_boundary_mask=left_context_boundary_mask,
        left_context_boundary_indices=left_context_boundary_indices,
        left_context_reuse_segments=left_context_reuse_segments,
        candidate_positions=candidate_positions,
        candidate_position_max=int(candidate_plan.candidate_position_max),
        candidate_sample_positions=candidate_sample_positions,
        candidate_sample_position_max=int(candidate_plan.candidate_sample_position_max),
        compact_candidate_input=bool(compact_candidate_input),
        batch_shared_sequence=bool(batch_shared_sequence),
    )
    selected_indices, metadata = _build_compressed_kv_selection(
        candidate_plan,
        local_valid_mask,
        local_block_starts,
        cp_size,
        cp_rank,
        candidate_capacity,
        output_size,
        compression_ratio,
        local_seq_len,
        local_seq_offset,
        total_seq_len,
        local_kv.device,
        batch_shared_sequence,
    )
    return DeepSeekV4CPCompressionCandidates(
        candidate_blocks=candidate_blocks,
        compress_context=compress_context,
        selected_indices=selected_indices,
        metadata=metadata,
        cp_group=cp_group,
        output_size=int(output_size),
    )


def _resolve_batch_shared_sequence_cu_seqlens(
    local_seq_len,
    cp_size,
    cu_seqlens,
    batch_size,
):
    """Collapse fixed BSND batch boundaries to one shared sequence plan."""

    total_seq_len = int(local_seq_len) * int(cp_size)
    shared_cu_seqlens = (0, total_seq_len)
    if cu_seqlens is None:
        return total_seq_len, shared_cu_seqlens

    boundaries = _to_int_list(cu_seqlens)
    if len(boundaries) < 2 or boundaries[0] != 0:
        raise ValueError("batch-shared BSND cu_seqlens must start with 0 and contain at least two entries.")
    expected_boundary_count = int(batch_size) + 1
    if len(boundaries) != expected_boundary_count:
        raise ValueError(
            "batch-shared BSND cu_seqlens must contain one boundary per batch: "
            f"expected {expected_boundary_count} entries, got {len(boundaries)}."
        )
    lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
    if any(length < 0 for length in lengths):
        raise ValueError("batch-shared BSND cu_seqlens must be monotonically non-decreasing.")
    if any(length != total_seq_len for length in lengths):
        raise ValueError(
            "batch-shared BSND cu_seqlens must describe equal global sequence lengths: "
            f"expected every length to be {total_seq_len}, got {lengths}."
        )
    return total_seq_len, shared_cu_seqlens


def _can_use_compact_candidate_view(
    requested,
    seq_first_input_is_contiguous,
    local_seq_offset,
    local_seq_len,
    compression_ratio,
    total_seq_len,
    candidate_plan,
    previous_tail_len,
    include_left_context,
):
    """Return whether fixed aligned candidates can alias the local source."""

    if not requested or not seq_first_input_is_contiguous:
        return False
    if local_seq_len <= 0:
        return False
    if int(local_seq_offset) % int(compression_ratio) != 0 or int(local_seq_len) % int(compression_ratio) != 0:
        return False

    sample_boundaries = candidate_plan.sample_boundaries
    if (
        sample_boundaries.numel() != 2
        or int(sample_boundaries[0]) != 0
        or int(sample_boundaries[1]) != int(total_seq_len)
    ):
        return False

    expected_candidate_count = int(local_seq_len) // int(compression_ratio)
    if candidate_plan.candidate_count != expected_candidate_count:
        return False
    expected_starts = torch.arange(
        int(local_seq_offset),
        int(local_seq_offset) + int(local_seq_len),
        int(compression_ratio),
        dtype=torch.long,
    )
    actual_starts = candidate_plan.candidate_starts[:expected_candidate_count]
    if not torch.equal(actual_starts, expected_starts):
        return False

    if not include_left_context:
        return True
    boundary_indices = candidate_plan.left_context_boundary_indices
    if boundary_indices.numel() > 1:
        return False
    if boundary_indices.numel() == 0:
        return True
    boundary_offsets = candidate_plan.left_context_offsets.index_select(
        0,
        boundary_indices,
    )
    return bool(
        torch.all(boundary_offsets >= 0)
        and torch.all(boundary_offsets + int(compression_ratio) <= int(previous_tail_len))
    )


def _view_compact_left_context_boundaries(
    previous_tail,
    candidate_plan,
    compression_ratio,
):
    """Return only cross-rank left blocks, aliasing previous-tail storage."""

    boundary_indices = candidate_plan.left_context_boundary_indices
    boundary_count = int(boundary_indices.numel())
    output_shape = (
        boundary_count,
        int(compression_ratio),
        *previous_tail.shape[1:],
    )
    if boundary_count == 0:
        return previous_tail.narrow(0, 0, 0).reshape(output_shape)
    if boundary_count != 1:
        raise RuntimeError("compact candidate view supports at most one left-context boundary.")

    boundary_offset = int(candidate_plan.left_context_offsets[int(boundary_indices[0])])
    return previous_tail.narrow(
        0,
        boundary_offset,
        int(compression_ratio),
    ).unsqueeze(0)


def _mask_and_pad_local_compressed(
    local_compressed,
    candidate_blocks,
    compress_context,
):
    candidate_count = int(candidate_blocks.shape[0])
    if local_compressed.shape[0] != candidate_count:
        raise ValueError(
            "local compressed KV must preserve the candidate_count dimension: "
            f"expected {candidate_count}, got {local_compressed.shape[0]}"
        )
    if bool(getattr(compress_context, "batch_shared_sequence", False)):
        expected_batch_size = int(candidate_blocks.shape[2])
        if local_compressed.dim() < 2 or int(local_compressed.shape[1]) != expected_batch_size:
            actual_batch_size = int(local_compressed.shape[1]) if local_compressed.dim() >= 2 else None
            raise ValueError(
                "local compressed KV must preserve the BSND batch dimension after candidate compression: "
                f"expected {expected_batch_size}, got {actual_batch_size}."
            )

    valid_mask = compress_context.valid_mask
    if valid_mask.shape[0] != candidate_count:
        raise ValueError(
            "compress context valid_mask must match the candidate count: "
            f"expected {candidate_count}, got {valid_mask.shape[0]}."
        )
    local_compressed = torch.where(
        _view_as_broadcast_mask(valid_mask, local_compressed).to(dtype=torch.bool),
        local_compressed,
        torch.zeros_like(local_compressed),
    )

    candidate_capacity = int(compress_context.candidate_capacity)
    if not bool(getattr(compress_context, "compact_candidate_input", False)):
        if candidate_count != candidate_capacity:
            raise ValueError(
                "non-compact candidate input must match candidate_capacity: "
                f"expected {candidate_capacity}, got {candidate_count}."
            )
        return local_compressed
    if candidate_count > candidate_capacity:
        raise ValueError(
            "compact candidate count exceeds candidate_capacity: "
            f"capacity={candidate_capacity}, count={candidate_count}."
        )

    padding_count = candidate_capacity - candidate_count
    if padding_count == 0:
        return local_compressed
    padding = local_compressed.new_zeros((padding_count,) + tuple(local_compressed.shape[1:]))
    return torch.cat((local_compressed, padding), dim=0)


def launch_deepseek_v4_allgather_compressed_kv(
    prepared_candidates: DeepSeekV4CPCompressionCandidates,
    local_compressed: torch.Tensor,
) -> DeepSeekV4CPPendingCompressedKV:
    """Mask/pad local compressed KV and launch all-gather without waiting."""
    if not isinstance(prepared_candidates, DeepSeekV4CPCompressionCandidates):
        raise TypeError("prepared_candidates must be a DeepSeekV4CPCompressionCandidates instance.")
    if not torch.is_tensor(local_compressed):
        raise TypeError("local_compressed must be a torch.Tensor.")

    compress_context = prepared_candidates.compress_context
    local_compressed = _mask_and_pad_local_compressed(
        local_compressed,
        prepared_candidates.candidate_blocks,
        compress_context,
    )
    collective_state = _AsyncCollectiveState()
    gathered_compressed = _AllGatherCompressedKVAsync.apply(
        local_compressed,
        prepared_candidates.cp_group,
        collective_state,
    )
    return DeepSeekV4CPPendingCompressedKV(
        gathered_compressed=gathered_compressed,
        selected_indices=prepared_candidates.selected_indices,
        metadata=prepared_candidates.metadata,
        output_size=prepared_candidates.output_size,
        seq_dim=compress_context.seq_dim,
        collective_state=collective_state,
    )


def wait_deepseek_v4_compressed_kv(
    pending: DeepSeekV4CPPendingCompressedKV,
) -> DeepSeekV4CPCompressedKV:
    """Wait for a launched compressed-KV all-gather and finalize global order."""
    if not isinstance(pending, DeepSeekV4CPPendingCompressedKV):
        raise TypeError("pending must be a DeepSeekV4CPPendingCompressedKV instance.")
    if pending.resolved is not None:
        return pending.resolved

    pending.collective_state.wait()
    compressed_kv = _select_and_pad_gathered_compressed(
        pending.gathered_compressed,
        pending.selected_indices,
        pending.output_size,
        pending.seq_dim,
    )
    pending.resolved = DeepSeekV4CPCompressedKV(
        compressed_kv=compressed_kv,
        metadata=pending.metadata,
    )
    return pending.resolved


def compact_deepseek_v4_compressed_kv(
    prepared_compressed_kv: DeepSeekV4CPCompressedKV,
    seq_dim: int = 0,
):
    """Remove padding from prepared compressed KV and return aligned block starts."""
    compressed_seq_first = torch.movedim(prepared_compressed_kv.compressed_kv, seq_dim, 0).contiguous()
    metadata = prepared_compressed_kv.metadata
    block_starts = metadata.block_starts.to(device=compressed_seq_first.device)
    if metadata.valid_count is None:
        valid_mask = metadata.valid_mask.to(device=compressed_seq_first.device)
        compact = compressed_seq_first[valid_mask]
        starts = block_starts[valid_mask]
    else:
        valid_count = int(metadata.valid_count)
        if valid_count < 0 or valid_count > compressed_seq_first.shape[0]:
            raise ValueError(
                "metadata.valid_count must be within the compressed KV sequence length: "
                f"valid_count={valid_count}, sequence_length={compressed_seq_first.shape[0]}."
            )
        compact = compressed_seq_first.narrow(0, 0, valid_count)
        starts = block_starts.narrow(0, 0, valid_count)
    return torch.movedim(compact, 0, seq_dim).contiguous(), starts


def _build_local_candidate_plan(
    local_seq_offset,
    local_seq_len,
    compression_ratio,
    cu_seqlens,
    total_seq_len,
    candidate_capacity,
    extended_start,
    extended_seq_len,
    include_left_context,
):
    """Build candidate and optional left-context metadata without device scalar updates."""

    candidate_starts, candidate_sample_ids, sample_boundaries = _enumerate_local_candidates(
        local_seq_offset,
        local_seq_len,
        compression_ratio,
        cu_seqlens,
        total_seq_len,
        candidate_capacity,
    )
    candidate_count = int(candidate_starts.numel())
    padded_starts = torch.full((candidate_capacity,), -1, dtype=torch.long)
    padded_offsets = torch.zeros(candidate_capacity, dtype=torch.long)
    padded_positions = torch.zeros(candidate_capacity, dtype=torch.long)
    padded_sample_positions = torch.zeros(candidate_capacity, dtype=torch.long)
    valid_mask = torch.zeros(candidate_capacity, dtype=torch.bool)
    candidate_position_max = 0
    candidate_sample_position_max = 0
    if candidate_count > 0:
        candidate_offsets = candidate_starts - int(extended_start)
        sample_starts = sample_boundaries[:-1].index_select(0, candidate_sample_ids)
        candidate_sample_positions = candidate_starts - sample_starts
        _validate_block_offsets(
            candidate_offsets,
            compression_ratio,
            extended_seq_len,
            "compressed-KV candidate",
            candidate_starts,
            extended_start,
        )
        padded_starts[:candidate_count] = candidate_starts
        padded_offsets[:candidate_count] = candidate_offsets
        padded_positions[:candidate_count] = candidate_starts
        padded_sample_positions[:candidate_count] = candidate_sample_positions
        valid_mask[:candidate_count] = True
        candidate_position_max = int(candidate_starts.max().item())
        candidate_sample_position_max = int(candidate_sample_positions.max().item())

    if not include_left_context:
        return _CandidatePlan(
            candidate_starts=padded_starts,
            candidate_offsets=padded_offsets,
            valid_mask=valid_mask,
            candidate_count=candidate_count,
            candidate_positions=padded_positions,
            candidate_position_max=candidate_position_max,
            candidate_sample_positions=padded_sample_positions,
            candidate_sample_position_max=candidate_sample_position_max,
            sample_boundaries=sample_boundaries,
        )

    left_context_starts = torch.full((candidate_capacity,), -1, dtype=torch.long)
    left_context_offsets = torch.zeros(candidate_capacity, dtype=torch.long)
    left_context_valid_mask = torch.zeros(candidate_capacity, dtype=torch.bool)
    padded_sample_ids = torch.full((candidate_capacity,), -1, dtype=torch.long)
    left_context_source_indices = torch.full(
        (candidate_capacity,),
        -1,
        dtype=torch.long,
    )
    left_context_boundary_mask = torch.zeros(
        candidate_capacity,
        dtype=torch.bool,
    )
    left_context_boundary_indices = torch.empty(0, dtype=torch.long)
    left_context_reuse_segments = ()
    if candidate_count > 0:
        unpadded_left_starts = candidate_starts - compression_ratio
        unpadded_left_valid_mask = unpadded_left_starts >= sample_starts
        unpadded_left_offsets = unpadded_left_starts - int(extended_start)
        _validate_block_offsets(
            unpadded_left_offsets[unpadded_left_valid_mask],
            compression_ratio,
            extended_seq_len,
            "left-context candidate",
            unpadded_left_starts[unpadded_left_valid_mask],
            extended_start,
        )
        left_context_starts[:candidate_count] = torch.where(
            unpadded_left_valid_mask,
            unpadded_left_starts,
            torch.full_like(unpadded_left_starts, -1),
        )
        left_context_offsets[:candidate_count] = torch.where(
            unpadded_left_valid_mask,
            unpadded_left_offsets,
            torch.zeros_like(unpadded_left_offsets),
        )
        left_context_valid_mask[:candidate_count] = unpadded_left_valid_mask
        padded_sample_ids[:candidate_count] = candidate_sample_ids
        (
            left_context_source_indices,
            left_context_boundary_mask,
            left_context_boundary_indices,
            left_context_reuse_segments,
        ) = _build_left_context_reuse_metadata(
            candidate_starts,
            candidate_sample_ids,
            unpadded_left_starts,
            unpadded_left_valid_mask,
            candidate_capacity,
        )

    return _CandidatePlan(
        candidate_starts=padded_starts,
        candidate_offsets=padded_offsets,
        valid_mask=valid_mask,
        candidate_count=candidate_count,
        candidate_positions=padded_positions,
        candidate_position_max=candidate_position_max,
        candidate_sample_positions=padded_sample_positions,
        candidate_sample_position_max=candidate_sample_position_max,
        sample_boundaries=sample_boundaries,
        left_context_starts=left_context_starts,
        left_context_offsets=left_context_offsets,
        left_context_valid_mask=left_context_valid_mask,
        sample_ids=padded_sample_ids,
        left_context_source_indices=left_context_source_indices,
        left_context_boundary_mask=left_context_boundary_mask,
        left_context_boundary_indices=left_context_boundary_indices,
        left_context_reuse_segments=left_context_reuse_segments,
    )


def _build_left_context_reuse_metadata(
    candidate_starts,
    candidate_sample_ids,
    left_context_starts,
    left_context_valid_mask,
    candidate_capacity,
):
    """Map left-context rows to local candidate projections or compact boundaries."""

    source_indices = torch.full(
        (candidate_capacity,),
        -1,
        dtype=torch.long,
    )
    boundary_mask = torch.zeros(candidate_capacity, dtype=torch.bool)
    valid_targets = torch.nonzero(
        left_context_valid_mask,
        as_tuple=False,
    ).flatten()
    if valid_targets.numel() == 0:
        return (
            source_indices,
            boundary_mask,
            torch.empty(0, dtype=torch.long),
            (),
        )

    target_left_starts = left_context_starts.index_select(0, valid_targets)
    insertion_indices = torch.searchsorted(
        candidate_starts,
        target_left_starts,
        right=False,
    )
    safe_indices = insertion_indices.clamp(
        min=0,
        max=max(int(candidate_starts.numel()) - 1, 0),
    )
    source_matches = insertion_indices < candidate_starts.numel()
    source_matches &= candidate_starts.index_select(0, safe_indices) == target_left_starts
    source_matches &= candidate_sample_ids.index_select(0, safe_indices) == candidate_sample_ids.index_select(
        0, valid_targets
    )

    target_sources = torch.where(
        source_matches,
        safe_indices,
        torch.full_like(safe_indices, -1),
    )
    source_indices[valid_targets] = target_sources

    boundary_indices = valid_targets[target_sources < 0]
    boundary_mask[boundary_indices] = True

    reusable_mask = target_sources >= 0
    reusable_targets = valid_targets[reusable_mask]
    reusable_sources = target_sources[reusable_mask]
    if reusable_targets.numel() == 0:
        return source_indices, boundary_mask, boundary_indices, ()

    segment_breaks = torch.ones(
        reusable_targets.numel(),
        dtype=torch.bool,
    )
    if reusable_targets.numel() > 1:
        segment_breaks[1:] = (torch.diff(reusable_targets) != 1) | (torch.diff(reusable_sources) != 1)
    segment_offsets = (
        torch.nonzero(
            segment_breaks,
            as_tuple=False,
        )
        .flatten()
        .tolist()
    )
    segments = []
    for segment_index, segment_offset in enumerate(segment_offsets):
        next_offset = (
            segment_offsets[segment_index + 1] if segment_index + 1 < len(segment_offsets) else reusable_targets.numel()
        )
        target_start = int(reusable_targets[segment_offset].item())
        target_end = int(reusable_targets[next_offset - 1].item()) + 1
        source_start = int(reusable_sources[segment_offset].item())
        segments.append((target_start, target_end, source_start))

    return source_indices, boundary_mask, boundary_indices, tuple(segments)


def _enumerate_local_candidates(
    local_seq_offset,
    local_seq_len,
    compression_ratio,
    cu_seqlens,
    total_seq_len,
    candidate_capacity,
):
    """Enumerate only blocks owned by this rank instead of all global blocks."""

    local_start = int(local_seq_offset)
    local_end = local_start + int(local_seq_len)
    sample_boundaries = _get_sample_boundaries_tensor(cu_seqlens, total_seq_len, device="cpu")
    sample_starts = sample_boundaries[:-1]
    sample_lengths = torch.diff(sample_boundaries)
    block_counts = torch.div(sample_lengths, compression_ratio, rounding_mode="floor")
    if block_counts.numel() == 0:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            sample_boundaries,
        )

    first_block_numbers = (
        torch.div(
            local_start - sample_starts,
            compression_ratio,
            rounding_mode="floor",
        )
        + 1
    )
    first_block_numbers.clamp_(min=1)
    last_block_numbers = torch.div(
        local_end - sample_starts,
        compression_ratio,
        rounding_mode="floor",
    )
    last_block_numbers = torch.minimum(last_block_numbers, block_counts)
    local_block_counts = (last_block_numbers - first_block_numbers + 1).clamp_(min=0)
    candidate_count = int(local_block_counts.sum().item())
    if candidate_count > candidate_capacity:
        raise ValueError(
            f"Too many compressed KV candidates for this rank: capacity={candidate_capacity}, actual={candidate_count}."
        )
    if candidate_count == 0:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            sample_boundaries,
        )

    sample_ids = torch.repeat_interleave(
        torch.arange(sample_starts.numel(), dtype=torch.long),
        local_block_counts,
        output_size=candidate_count,
    )
    sample_candidate_offsets = torch.repeat_interleave(
        torch.cumsum(local_block_counts, dim=0) - local_block_counts,
        local_block_counts,
        output_size=candidate_count,
    )
    block_offsets = torch.arange(candidate_count, dtype=torch.long) - sample_candidate_offsets
    block_numbers = first_block_numbers.index_select(0, sample_ids) + block_offsets
    candidate_starts = sample_starts.index_select(0, sample_ids) + (block_numbers - 1) * compression_ratio
    return candidate_starts, sample_ids, sample_boundaries


def _validate_block_offsets(
    block_offsets,
    compression_ratio,
    extended_seq_len,
    description,
    block_starts,
    extended_start,
):
    if block_offsets.numel() == 0:
        return
    invalid_mask = (block_offsets < 0) | (block_offsets + compression_ratio > extended_seq_len)
    if not torch.any(invalid_mask):
        return
    invalid_index = int(torch.nonzero(invalid_mask, as_tuple=False)[0].item())
    invalid_start = int(block_starts[invalid_index].item())
    raise ValueError(
        f"{description} is outside the received previous-tail window: "
        f"block_start={invalid_start}, extended_start={extended_start}, "
        f"compression_ratio={compression_ratio}."
    )


def _gather_padded_blocks(
    extended_kv,
    block_offsets,
    valid_mask,
    compression_ratio,
):
    """Materialize all padded blocks with one index_select and one mask operation."""

    if block_offsets.shape != valid_mask.shape:
        raise ValueError(
            "block_offsets and valid_mask must have the same shape: "
            f"got {tuple(block_offsets.shape)} and {tuple(valid_mask.shape)}."
        )
    gather_indices = block_offsets.view(-1, 1) + torch.arange(
        compression_ratio,
        dtype=torch.long,
        device=extended_kv.device,
    )
    gathered = extended_kv.index_select(0, gather_indices.reshape(-1)).view(
        block_offsets.shape[0],
        compression_ratio,
        *extended_kv.shape[1:],
    )
    mask_shape = (valid_mask.shape[0],) + (1,) * (gathered.dim() - 1)
    return gathered * valid_mask.to(dtype=gathered.dtype).view(mask_shape)


def _view_as_broadcast_mask(mask, tensor):
    view_shape = (mask.shape[0],) + (1,) * (tensor.dim() - 1)
    return mask.to(dtype=tensor.dtype).view(view_shape)


_DETERMINISTIC_SELECTION_CACHE_CAPACITY = 16
_deterministic_selection_cache = OrderedDict()


def _build_compressed_kv_selection(
    candidate_plan,
    local_valid_mask,
    local_block_starts,
    cp_size,
    cp_rank,
    candidate_capacity,
    output_size,
    compression_ratio,
    local_seq_len,
    local_seq_offset,
    total_seq_len,
    device,
    batch_shared_sequence=False,
):
    if not _uses_contiguous_rank_order_partition(
        cp_size,
        cp_rank,
        local_seq_len,
        local_seq_offset,
        total_seq_len,
    ):
        raise ValueError(
            "DeepSeek V4 CP compressed-KV selection requires equal contiguous rank-order sequence partitions."
        )

    plan = _get_deterministic_selection_plan(
        candidate_plan.sample_boundaries,
        cp_size,
        candidate_capacity,
        output_size,
        compression_ratio,
        local_seq_len,
        device,
    )
    _validate_local_candidates_against_selection_plan(
        candidate_plan,
        plan,
        cp_rank,
    )
    metadata = DeepSeekV4CPMetadata(
        valid_mask=plan.valid_mask,
        block_starts=plan.block_starts,
        source_rank=plan.source_rank,
        local_valid_mask=local_valid_mask,
        local_block_starts=local_block_starts,
        compression_ratio=compression_ratio,
        local_seq_len=local_seq_len,
        output_size=output_size,
        valid_count=plan.valid_count,
        is_identity_compact_order=plan.is_identity_compact_order,
        batch_shared_sequence=bool(batch_shared_sequence),
    )
    return plan.selected_indices, metadata


def _uses_contiguous_rank_order_partition(
    cp_size,
    cp_rank,
    local_seq_len,
    local_seq_offset,
    total_seq_len,
):
    return int(total_seq_len) == int(local_seq_len) * int(cp_size) and int(local_seq_offset) == int(
        local_seq_len
    ) * int(cp_rank)


def _get_deterministic_selection_plan(
    sample_boundaries,
    cp_size,
    candidate_capacity,
    output_size,
    compression_ratio,
    local_seq_len,
    device,
):
    cache_key = (
        tuple(int(value) for value in sample_boundaries.tolist()),
        int(cp_size),
        int(candidate_capacity),
        int(output_size),
        int(compression_ratio),
        int(local_seq_len),
        device.type,
        device.index,
    )
    cached = _deterministic_selection_cache.pop(cache_key, None)
    if cached is not None:
        _deterministic_selection_cache[cache_key] = cached
        return cached

    plan = _build_deterministic_selection_plan(
        sample_boundaries,
        cp_size,
        candidate_capacity,
        output_size,
        compression_ratio,
        local_seq_len,
        device,
    )
    _deterministic_selection_cache[cache_key] = plan
    while len(_deterministic_selection_cache) > _DETERMINISTIC_SELECTION_CACHE_CAPACITY:
        _deterministic_selection_cache.popitem(last=False)
    return plan


def _build_deterministic_selection_plan(
    sample_boundaries,
    cp_size,
    candidate_capacity,
    output_size,
    compression_ratio,
    local_seq_len,
    device,
):
    global_block_starts = enumerate_global_block_starts(
        sample_boundaries,
        compression_ratio,
    )
    candidate_count = int(global_block_starts.numel())
    if candidate_count > output_size:
        raise ValueError(
            f"Compressed KV candidates exceed output_size: candidates={candidate_count}, output_size={output_size}."
        )

    if candidate_count > 0:
        source_rank = torch.div(
            global_block_starts + compression_ratio - 1,
            local_seq_len,
            rounding_mode="floor",
        )
        if int(source_rank.min()) < 0 or int(source_rank.max()) >= cp_size:
            raise ValueError("Compressed KV candidate owner is outside the contiguous CP partition.")
        rank_candidate_counts = torch.bincount(source_rank, minlength=cp_size)
        if int(rank_candidate_counts.max()) > candidate_capacity:
            raise ValueError(
                "Too many compressed KV candidates for a contiguous CP rank: "
                f"capacity={candidate_capacity}, actual={int(rank_candidate_counts.max())}."
            )
        rank_candidate_offsets = torch.cumsum(rank_candidate_counts, dim=0) - rank_candidate_counts
        local_candidate_ordinals = torch.arange(
            candidate_count, dtype=torch.long
        ) - rank_candidate_offsets.index_select(0, source_rank)
        selected_indices = source_rank * candidate_capacity + local_candidate_ordinals
    else:
        source_rank = torch.empty(0, dtype=torch.long)
        selected_indices = torch.empty(0, dtype=torch.long)
        rank_candidate_counts = torch.zeros(cp_size, dtype=torch.long)

    valid_mask = torch.zeros(output_size, dtype=torch.bool)
    padded_block_starts = torch.full((output_size,), -1, dtype=torch.long)
    padded_source_rank = torch.full((output_size,), -1, dtype=torch.long)
    if candidate_count > 0:
        valid_mask[:candidate_count] = True
        padded_block_starts[:candidate_count] = global_block_starts
        padded_source_rank[:candidate_count] = source_rank

    rank_candidate_starts = torch.split(
        global_block_starts,
        tuple(int(count) for count in rank_candidate_counts.tolist()),
    )
    return _DeterministicSelectionPlan(
        selected_indices=selected_indices.to(device=device),
        valid_count=candidate_count,
        valid_mask=valid_mask.to(device=device),
        block_starts=padded_block_starts.to(device=device),
        source_rank=padded_source_rank.to(device=device),
        rank_candidate_starts=rank_candidate_starts,
        is_identity_compact_order=(sample_boundaries.numel() == 2 and int(sample_boundaries[0]) == 0),
    )


def _validate_local_candidates_against_selection_plan(
    candidate_plan,
    selection_plan,
    cp_rank,
):
    local_candidate_starts = candidate_plan.candidate_starts[: candidate_plan.candidate_count]
    expected_candidate_starts = selection_plan.rank_candidate_starts[cp_rank]
    if not torch.equal(local_candidate_starts, expected_candidate_starts):
        raise RuntimeError("Local compressed-KV candidates do not match the contiguous rank-order selection plan.")


def _select_and_pad_gathered_compressed(
    gathered_compressed,
    selected_indices,
    output_size,
    seq_dim,
):
    selected = (
        gathered_compressed.index_select(0, selected_indices)
        if selected_indices.numel() > 0
        else gathered_compressed.new_empty((0,) + tuple(gathered_compressed.shape[1:]))
    )
    pad_len = output_size - selected.shape[0]
    if pad_len < 0:
        raise ValueError(
            f"Compressed KV candidates exceed output_size: candidates={selected.shape[0]}, output_size={output_size}."
        )
    if pad_len > 0:
        padding = gathered_compressed.new_zeros((pad_len,) + tuple(gathered_compressed.shape[1:]))
        selected = torch.cat((selected, padding), dim=0)
    return torch.movedim(selected, 0, seq_dim).contiguous()
