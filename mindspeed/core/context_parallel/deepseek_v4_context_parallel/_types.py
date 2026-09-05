# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import torch


class _AsyncCollectiveStateProtocol(Protocol):
    completed: bool

    def wait(self) -> None: ...


@dataclass
class DeepSeekV4CPMetadata:
    valid_mask: torch.Tensor
    block_starts: torch.Tensor
    source_rank: torch.Tensor
    local_valid_mask: torch.Tensor
    local_block_starts: torch.Tensor
    compression_ratio: int
    local_seq_len: int
    output_size: int
    valid_count: Optional[int] = None
    is_identity_compact_order: bool = False
    batch_shared_sequence: bool = False


@dataclass
class DeepSeekV4CPCompressedKV:
    compressed_kv: torch.Tensor
    metadata: DeepSeekV4CPMetadata


@dataclass
class DeepSeekV4CPPendingCompressedKV:
    """Compressed KV whose cross-rank all-gather has been launched asynchronously."""

    gathered_compressed: torch.Tensor
    selected_indices: torch.Tensor
    metadata: DeepSeekV4CPMetadata
    output_size: int
    seq_dim: int
    collective_state: _AsyncCollectiveStateProtocol
    resolved: Optional[DeepSeekV4CPCompressedKV] = None


@dataclass
class DeepSeekV4CPPackedSeqMetadata:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_ori_kv: torch.Tensor
    cu_seqlens: torch.Tensor
    query_positions: torch.Tensor
    local_seq_offset: int


@dataclass
class DeepSeekV4CPCompressContext:
    """Context passed to model-owned compressed-KV callbacks.

    MindSpeed owns CP communication, block selection, and the compressor
    adapter. Callers may use this context to align RoPE, compressor state,
    or sample boundaries.
    """

    candidate_starts: torch.Tensor
    valid_mask: torch.Tensor
    local_seq_offset: int
    local_seq_len: int
    total_seq_len: int
    compression_ratio: int
    candidate_capacity: int
    cp_size: int
    cp_rank: int
    seq_dim: int
    cu_seqlens: Optional[Sequence[int]]
    left_context_blocks: Optional[torch.Tensor] = None
    left_context_boundary_blocks: Optional[torch.Tensor] = None
    left_context_starts: Optional[torch.Tensor] = None
    left_context_valid_mask: Optional[torch.Tensor] = None
    sample_ids: Optional[torch.Tensor] = None
    left_context_source_indices: Optional[torch.Tensor] = None
    left_context_boundary_mask: Optional[torch.Tensor] = None
    left_context_boundary_indices: Optional[torch.Tensor] = None
    left_context_reuse_segments: Optional[Sequence[tuple]] = None
    candidate_positions: Optional[torch.Tensor] = None
    candidate_position_max: Optional[int] = None
    candidate_sample_positions: Optional[torch.Tensor] = None
    candidate_sample_position_max: Optional[int] = None
    compact_candidate_input: bool = False
    batch_shared_sequence: bool = False


@dataclass
class DeepSeekV4CPCompressionCandidates:
    """Reusable CP candidate state for model-owned compression branches."""

    candidate_blocks: torch.Tensor
    compress_context: DeepSeekV4CPCompressContext
    selected_indices: torch.Tensor
    metadata: DeepSeekV4CPMetadata
    cp_group: object
    output_size: int


@dataclass
class DeepSeekV4CPAlignmentDescriptor:
    """Shared mapping from global compact blocks to operator-facing prefixes."""

    full_block_starts: torch.Tensor
    selected_global_indices: torch.Tensor
    global_to_local: torch.Tensor
    block_starts: torch.Tensor
    cu_seqlens_cmp_kv: torch.Tensor
    cmp_residual_kv: torch.Tensor
    is_identity_prefix: bool = False


@dataclass
class DeepSeekV4CPSMLAInputs:
    """Operator-facing tensors after local query-offset semantics are encoded.

    ``cmp_kv`` contains a per-sample compressed prefix ending at the last local
    query position. ``cu_seqlens_cmp_kv`` and ``cmp_residual_kv`` describe that
    prefix so SparseFlashMla's right-down causal mask aligns with the CP shard.
    """

    q: torch.Tensor
    ori_kv: torch.Tensor
    cmp_kv: Optional[torch.Tensor]
    cmp_sparse_indices: Optional[torch.Tensor]
    cu_seqlens_q: Optional[torch.Tensor]
    cu_seqlens_ori_kv: Optional[torch.Tensor]
    cu_seqlens_cmp_kv: Optional[torch.Tensor]
    seqused_ori_kv: Optional[torch.Tensor]
    seqused_cmp_kv: Optional[torch.Tensor]
    cmp_residual_kv: Optional[torch.Tensor]
    metadata: Optional[torch.Tensor]
    block_starts: Optional[torch.Tensor]
    alignment: Optional[DeepSeekV4CPAlignmentDescriptor] = None


@dataclass
class DeepSeekV4CPRuntimeMetadata:
    """Validated runtime metadata derived before invoking SparseFlashMla.

    ``cu_seqlens_ori_kv`` is the SMLA-facing cu_seqlens (matches the actual
    ori_kv tensor length — window length in CP window model).
    ``cu_seqlens_ori_kv_global`` is the global sequence boundary used by the
    compressed-KV preparation (block range / allgather / compact). In non-CP
    or full-prefix mode they are identical; in CP window model they differ.
    """

    query_positions: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_ori_kv: torch.Tensor
    cu_seqlens_ori_kv_global: torch.Tensor
    cu_seqlens_cmp_kv: Optional[torch.Tensor]
    seqused_ori_kv: Optional[torch.Tensor]
    seqused_cmp_kv: Optional[torch.Tensor]
    cmp_residual_kv: Optional[torch.Tensor]
    q_seq_dim: int
    ori_kv_seq_dim: int
    cmp_seq_dim: int
    local_seq_offset: int
    layout_q: str
    layout_kv: str
    batch_size: int
