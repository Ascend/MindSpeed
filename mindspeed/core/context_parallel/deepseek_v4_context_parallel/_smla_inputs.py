# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# pylint: disable=too-many-lines

from typing import Optional

import torch

from ._types import (
    DeepSeekV4CPAlignmentDescriptor,
    DeepSeekV4CPCompressedKV,
    DeepSeekV4CPPackedSeqMetadata,
    DeepSeekV4CPRuntimeMetadata,
    DeepSeekV4CPSMLAInputs,
)
from ._utils import (
    _build_deepseek_v4_cmp_visibility,
    _lengths_from_cu_seqlens,
    _reshape_deepseek_v4_sparse_indices_for_layout,
    _to_int32_tensor,
    normalize_cu_seqlens,
)
from ._validation import (
    _get_deepseek_v4_batch_size,
    _resolve_deepseek_v4_seq_dim,
)
from ._compressed_kv import compact_deepseek_v4_compressed_kv


def build_deepseek_v4_cmp_cu_seqlens(cu_seqlens, compression_ratio: int, device=None) -> torch.Tensor:
    """Build compressed-side cu_seqlens using floor(sample_len / compression_ratio)."""
    if compression_ratio <= 0:
        raise ValueError("compression_ratio must be a positive integer.")

    cu_seqlens = _to_int32_tensor(cu_seqlens, device)
    lengths = _lengths_from_cu_seqlens(cu_seqlens)
    compressed_tail = torch.cumsum(lengths // compression_ratio, dim=0, dtype=torch.int32)
    return torch.cat(
        (torch.zeros(1, dtype=torch.int32, device=cu_seqlens.device), compressed_tail),
        dim=0,
    )


def build_deepseek_v4_cmp_residual_kv(cu_seqlens, compression_ratio: int, device=None) -> torch.Tensor:
    """Build per-sample compressed KV residual lengths for SMLA mask recovery."""
    if compression_ratio <= 0:
        raise ValueError("compression_ratio must be a positive integer.")

    cu_seqlens = _to_int32_tensor(cu_seqlens, device)
    return _lengths_from_cu_seqlens(cu_seqlens) % compression_ratio


def build_deepseek_v4_cp_packed_seq_metadata(
    cu_seqlens,
    local_seq_offset: int,
    local_seq_len: int,
    device=None,
) -> DeepSeekV4CPPackedSeqMetadata:
    """Build TND metadata for a flattened CP shard of packed sequences.

    The CP shard is a single contiguous range in the global flattened packed
    sequence: ``[local_seq_offset, local_seq_offset + local_seq_len)``.  The
    returned local ``cu_seqlens_q`` keeps one entry per global sample, including
    zero-length samples, so SMLA-facing Q/Ori/Cmp cu_seqlens share the same
    batch cardinality.
    """
    if local_seq_len < 0:
        raise ValueError("local_seq_len must be non-negative.")
    local_seq_offset = int(local_seq_offset)
    local_seq_len = int(local_seq_len)

    cu_global = normalize_cu_seqlens(
        cu_seqlens,
        device,
        name="cu_seqlens",
        require_leading_zero=True,
    )
    local_start = local_seq_offset
    local_end = local_start + local_seq_len
    total_len = int(cu_global[-1].item())
    if local_start < 0 or local_end > total_len:
        raise ValueError(
            "local flattened CP shard must be inside global cu_seqlens: "
            f"range=[{local_start}, {local_end}), total={total_len}."
        )

    sample_starts = cu_global[:-1].to(dtype=torch.long)
    sample_ends = cu_global[1:].to(dtype=torch.long)
    overlap_starts = torch.maximum(
        sample_starts,
        torch.full_like(sample_starts, local_start),
    )
    overlap_ends = torch.minimum(
        sample_ends,
        torch.full_like(sample_ends, local_end),
    )
    local_lens = torch.clamp(overlap_ends - overlap_starts, min=0).to(dtype=torch.int32)
    local_cu = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=cu_global.device),
            torch.cumsum(local_lens, dim=0, dtype=torch.int32),
        ),
        dim=0,
    )
    if int(local_cu[-1].item()) != local_seq_len:
        raise ValueError(
            "packed local cu_seqlens_q does not match local_seq_len: "
            f"expected {local_seq_len}, got {int(local_cu[-1].item())}."
        )

    query_positions = torch.arange(
        local_start,
        local_end,
        dtype=torch.long,
        device=cu_global.device,
    )
    return DeepSeekV4CPPackedSeqMetadata(
        cu_seqlens_q=local_cu,
        cu_seqlens_ori_kv=local_cu.clone(),
        cu_seqlens=cu_global,
        query_positions=query_positions,
        local_seq_offset=local_start,
    )


def build_deepseek_v4_owned_runtime_metadata(
    query: torch.Tensor,
    layout: str,
    cu_seqlens_q,
    cu_seqlens_ori_kv,
    cu_seqlens_global,
    query_positions: torch.Tensor,
    local_seq_offset: int,
    batch_size: int,
    compression_ratio: int,
    q_seq_dim: Optional[int] = None,
    ori_kv_seq_dim: Optional[int] = None,
    cmp_seq_dim: Optional[int] = None,
) -> DeepSeekV4CPRuntimeMetadata:
    """Assemble Attention-owned runtime metadata without re-validating tensors."""
    device = query.device
    q_seq_dim = _resolve_deepseek_v4_seq_dim(layout, q_seq_dim, "q_seq_dim")
    ori_kv_seq_dim = _resolve_deepseek_v4_seq_dim(layout, ori_kv_seq_dim, "ori_kv_seq_dim")
    cmp_seq_dim = _resolve_deepseek_v4_seq_dim(layout, cmp_seq_dim, "cmp_seq_dim")
    cu_q = _to_int32_tensor(cu_seqlens_q, device)
    cu_ori = _to_int32_tensor(cu_seqlens_ori_kv, device)
    cu_global = _to_int32_tensor(cu_seqlens_global, device)
    positions = query_positions.to(device=device, dtype=torch.long).reshape(-1)

    cu_seqlens_cmp_kv = None
    cmp_residual_kv = None
    if compression_ratio > 1:
        cu_seqlens_cmp_kv = build_deepseek_v4_cmp_cu_seqlens(
            cu_global,
            compression_ratio,
            device=device,
        )
        cmp_residual_kv = build_deepseek_v4_cmp_residual_kv(
            cu_global,
            compression_ratio,
            device=device,
        )

    return DeepSeekV4CPRuntimeMetadata(
        query_positions=positions,
        cu_seqlens_q=cu_q,
        cu_seqlens_ori_kv=cu_ori,
        cu_seqlens_ori_kv_global=cu_global,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_ori_kv=None,
        seqused_cmp_kv=None,
        cmp_residual_kv=cmp_residual_kv,
        q_seq_dim=q_seq_dim,
        ori_kv_seq_dim=ori_kv_seq_dim,
        cmp_seq_dim=cmp_seq_dim,
        local_seq_offset=int(local_seq_offset),
        layout_q=layout,
        layout_kv=layout,
        batch_size=int(batch_size),
    )


def build_deepseek_v4_causal_cmp_sparse_indices(
    query_positions: torch.Tensor,
    block_starts: torch.Tensor,
    valid_mask: torch.Tensor,
    compression_ratio: int,
    cu_seqlens=None,
    sparse_count: int = 512,
) -> torch.Tensor:
    """Build padded compressed-block indices visible to each query token."""
    if sparse_count <= 0:
        raise ValueError("sparse_count must be a positive integer.")
    if compression_ratio <= 0:
        raise ValueError("compression_ratio must be a positive integer.")

    query_positions = query_positions.to(dtype=torch.long)
    block_starts = block_starts.to(device=query_positions.device, dtype=torch.long)
    valid_mask = valid_mask.to(device=query_positions.device, dtype=torch.bool)
    visibility = _build_deepseek_v4_cmp_visibility(
        query_positions,
        block_starts,
        valid_mask,
        compression_ratio,
        cu_seqlens,
    )
    query_count, block_count = visibility.shape
    sparse_indices = torch.full(
        (query_count, sparse_count),
        -1,
        dtype=torch.int32,
        device=query_positions.device,
    )
    if block_count == 0:
        return sparse_indices

    select_count = min(sparse_count, block_count)
    candidate_indices = torch.arange(block_count, dtype=torch.int32, device=query_positions.device).view(1, -1)
    masked_indices = torch.where(
        visibility,
        candidate_indices.expand(query_count, -1),
        candidate_indices.new_full((query_count, block_count), block_count),
    )
    selected = torch.sort(masked_indices, dim=-1).values[:, :select_count]
    selected = torch.where(selected < block_count, selected, selected.new_full(selected.shape, -1))
    sparse_indices[:, :select_count] = selected

    return sparse_indices


def _flatten_deepseek_v4_cmp_sparse_indices(cmp_sparse_indices, q, layout_q):
    if cmp_sparse_indices is None:
        return None
    if not torch.is_tensor(cmp_sparse_indices):
        raise TypeError("cmp_sparse_indices must be a torch.Tensor.")
    if cmp_sparse_indices.dtype != torch.int32:
        raise ValueError("cmp_sparse_indices must use dtype torch.int32.")

    if layout_q == "BSND":
        batch_size, seq_len = int(q.shape[0]), int(q.shape[1])
        query_count = batch_size * seq_len
        if cmp_sparse_indices.dim() == 4 and cmp_sparse_indices.shape[2] == 1:
            cmp_sparse_indices = cmp_sparse_indices.squeeze(2)
        if cmp_sparse_indices.dim() == 3:
            if cmp_sparse_indices.shape[0] == batch_size and cmp_sparse_indices.shape[1] == seq_len:
                return cmp_sparse_indices.reshape(query_count, cmp_sparse_indices.shape[-1])
            if cmp_sparse_indices.shape[0] == query_count and cmp_sparse_indices.shape[1] == 1:
                return cmp_sparse_indices.squeeze(1)
        if cmp_sparse_indices.dim() == 2 and cmp_sparse_indices.shape[0] == query_count:
            return cmp_sparse_indices
        raise ValueError(
            "cmp_sparse_indices for BSND layout must have shape [B, S, K], [B, S, 1, K], [B*S, K], or [B*S, 1, K]."
        )

    query_count = int(q.shape[0])
    if cmp_sparse_indices.dim() == 3 and cmp_sparse_indices.shape[1] == 1:
        cmp_sparse_indices = cmp_sparse_indices.squeeze(1)
    if cmp_sparse_indices.dim() == 2 and cmp_sparse_indices.shape[0] == query_count:
        return cmp_sparse_indices
    raise ValueError("cmp_sparse_indices for TND layout must have shape [T, K] or [T, 1, K].")


def validate_deepseek_v4_c4a_cmp_sparse_indices(
    cmp_sparse_indices: torch.Tensor,
    query_positions: torch.Tensor,
    block_starts: torch.Tensor,
    valid_mask: torch.Tensor,
    compression_ratio: int,
    cu_seqlens=None,
) -> torch.Tensor:
    """Validate compact-aligned C4A sparse block indices.

    ``cmp_sparse_indices`` must use compact compressed-KV block positions after
    ``compact_deepseek_v4_compressed_kv`` filtering. Non-padding values select
    columns in the visibility matrix derived from compact ``block_starts``.
    """
    if not torch.is_tensor(cmp_sparse_indices):
        raise TypeError("cmp_sparse_indices must be a torch.Tensor.")
    if cmp_sparse_indices.dtype != torch.int32:
        raise ValueError("cmp_sparse_indices must use dtype torch.int32.")
    if cmp_sparse_indices.dim() != 2:
        raise ValueError("flattened cmp_sparse_indices must have shape [query_count, topk].")
    topk = int(cmp_sparse_indices.shape[-1])
    if topk not in (512, 1024):
        raise ValueError("cmp_sparse_indices topk must be 512 or 1024.")

    visibility = _build_deepseek_v4_cmp_visibility(
        query_positions,
        block_starts,
        valid_mask,
        compression_ratio,
        cu_seqlens,
    )
    if int(cmp_sparse_indices.shape[0]) != int(visibility.shape[0]):
        raise ValueError(
            "cmp_sparse_indices query dimension must match query_positions: "
            f"expected {int(visibility.shape[0])}, got {int(cmp_sparse_indices.shape[0])}."
        )

    block_count = int(visibility.shape[1])
    non_padding = cmp_sparse_indices >= 0
    if block_count == 0:
        if torch.any(non_padding):
            raise ValueError("cmp_sparse_indices selects blocks, but compact compressed KV has no valid blocks.")
        return cmp_sparse_indices

    out_of_range = non_padding & (cmp_sparse_indices >= block_count)
    if torch.any(out_of_range):
        invalid_value = int(cmp_sparse_indices[out_of_range][0].item())
        raise ValueError(
            f"cmp_sparse_indices must use compact block indices in range [0, {block_count}); got {invalid_value}."
        )

    gather_indices = cmp_sparse_indices.clamp(min=0, max=block_count - 1).to(dtype=torch.long)
    selected_visibility = visibility.gather(1, gather_indices)
    invisible = non_padding & ~selected_visibility
    if torch.any(invisible):
        row = int(torch.nonzero(invisible, as_tuple=False)[0, 0].item())
        col = int(torch.nonzero(invisible, as_tuple=False)[0, 1].item())
        value = int(cmp_sparse_indices[row, col].item())
        raise ValueError(
            "cmp_sparse_indices contains a block that is not causally/sample visible: "
            f"row={row}, topk={col}, compact_index={value}."
        )
    missing_visible_selection = ~non_padding.any(dim=-1) & visibility.any(dim=-1)
    if torch.any(missing_visible_selection):
        row = int(torch.nonzero(missing_visible_selection, as_tuple=False)[0, 0].item())
        raise ValueError(
            f"cmp_sparse_indices contains an all-padding row despite causally visible compressed blocks: row={row}."
        )
    return cmp_sparse_indices


def _get_deepseek_v4_bsnd_shared_query_metadata(
    runtime_metadata: DeepSeekV4CPRuntimeMetadata,
    device,
):
    cu_q = runtime_metadata.cu_seqlens_q.to(device=device, dtype=torch.long)
    cu_global = runtime_metadata.cu_seqlens_ori_kv_global.to(device=device, dtype=torch.long)
    if cu_q.numel() != cu_global.numel():
        raise ValueError("BSND Q and global cu_seqlens must describe the same fixed-length batches.")

    q_lengths = torch.diff(cu_q)
    global_lengths = torch.diff(cu_global)
    batch_size = int(runtime_metadata.batch_size)
    if q_lengths.numel() != batch_size or global_lengths.numel() != batch_size:
        raise ValueError("BSND cu_seqlens batch cardinality must match the tensor batch dimension.")
    if q_lengths.numel() == 0:
        raise ValueError("BSND requires a positive batch size.")
    if not bool((q_lengths == q_lengths[0]).all().item()):
        raise ValueError("BSND query sequences must have one shared fixed length.")
    if not bool((global_lengths == global_lengths[0]).all().item()):
        raise ValueError("BSND global sequences must have one shared fixed length.")

    query_positions = runtime_metadata.query_positions.to(device=device, dtype=torch.long)
    sample_indices = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.long, device=device),
        q_lengths,
    )
    if sample_indices.numel() != query_positions.numel():
        raise ValueError("BSND cu_seqlens_q must match the flattened query_positions length.")
    sample_starts = cu_global[:-1].index_select(0, sample_indices)
    local_query_positions = query_positions - sample_starts
    global_seq_len = int(global_lengths[0].item())
    if torch.any(local_query_positions < 0) or torch.any(local_query_positions >= global_seq_len):
        raise ValueError("BSND query_positions must stay inside each sample's global sequence.")

    q_seq_len = int(q_lengths[0].item())
    if q_seq_len > 0:
        local_query_matrix = local_query_positions.view(batch_size, q_seq_len)
        if not torch.equal(local_query_matrix, local_query_matrix[0].expand_as(local_query_matrix)):
            raise ValueError("BSND batches must use the same sample-local query positions on each CP rank.")
        if q_seq_len > 1 and not bool((torch.diff(local_query_matrix[0]) == 1).all().item()):
            raise ValueError("BSND query_positions must be contiguous inside each fixed-length sample.")
    return local_query_positions, q_seq_len, global_seq_len


def _build_deepseek_v4_bsnd_cp_alignment_descriptor(
    block_starts: torch.Tensor,
    runtime_metadata: DeepSeekV4CPRuntimeMetadata,
    compression_ratio: int,
    is_identity_compact_order: bool,
) -> DeepSeekV4CPAlignmentDescriptor:
    local_query_positions, q_seq_len, _ = _get_deepseek_v4_bsnd_shared_query_metadata(
        runtime_metadata,
        block_starts.device,
    )
    batch_size = int(runtime_metadata.batch_size)
    prefix_ori_len = int(local_query_positions[q_seq_len - 1].item()) + 1 if q_seq_len > 0 else 0
    prefix_cmp_len = prefix_ori_len // int(compression_ratio)
    prefix_residual = prefix_ori_len % int(compression_ratio)

    full_block_starts = block_starts.to(dtype=torch.long)
    block_count = int(full_block_starts.numel())
    expected_block_starts = torch.arange(
        0,
        prefix_cmp_len * int(compression_ratio),
        int(compression_ratio),
        dtype=torch.long,
        device=full_block_starts.device,
    )
    selected_mask = (
        (full_block_starts >= 0)
        & (full_block_starts < prefix_cmp_len * int(compression_ratio))
        & (torch.remainder(full_block_starts, int(compression_ratio)) == 0)
    )
    selected_global_indices = torch.nonzero(selected_mask, as_tuple=False).reshape(-1)
    selected_block_starts = full_block_starts.index_select(0, selected_global_indices)
    if selected_global_indices.numel() != prefix_cmp_len or not torch.equal(
        selected_block_starts,
        expected_block_starts,
    ):
        raise ValueError(
            "batch-shared BSND compressed KV does not contain the complete prefix required by local queries: "
            f"expected_blocks={prefix_cmp_len}, actual_blocks={selected_global_indices.numel()}."
        )

    local_indices = torch.arange(prefix_cmp_len, dtype=torch.long, device=full_block_starts.device)
    global_to_local = torch.full(
        (block_count,),
        -1,
        dtype=torch.long,
        device=full_block_starts.device,
    )
    if prefix_cmp_len > 0:
        global_to_local[selected_global_indices] = local_indices
    cu_seqlens_cmp_kv = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=full_block_starts.device) * prefix_cmp_len
    )
    cmp_residual_kv = torch.full(
        (batch_size,),
        prefix_residual,
        dtype=torch.int32,
        device=full_block_starts.device,
    )
    identity_indices = torch.arange(
        prefix_cmp_len,
        dtype=torch.long,
        device=full_block_starts.device,
    )
    is_identity_prefix = bool(is_identity_compact_order) and torch.equal(
        selected_global_indices,
        identity_indices,
    )
    return DeepSeekV4CPAlignmentDescriptor(
        full_block_starts=full_block_starts,
        selected_global_indices=selected_global_indices,
        global_to_local=global_to_local,
        block_starts=selected_block_starts,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        is_identity_prefix=is_identity_prefix,
    )


def _build_deepseek_v4_cp_alignment_descriptor(
    block_starts: torch.Tensor,
    runtime_metadata: DeepSeekV4CPRuntimeMetadata,
    compression_ratio: int,
    is_identity_compact_order: bool = False,
) -> DeepSeekV4CPAlignmentDescriptor:
    """Describe the compact prefix that right-aligns SMLA to local queries."""
    if runtime_metadata.layout_kv == "BSND":
        return _build_deepseek_v4_bsnd_cp_alignment_descriptor(
            block_starts,
            runtime_metadata,
            compression_ratio,
            is_identity_compact_order,
        )

    cu_q = runtime_metadata.cu_seqlens_q.to(device=block_starts.device, dtype=torch.long)
    cu_global = runtime_metadata.cu_seqlens_ori_kv_global.to(device=block_starts.device, dtype=torch.long)
    query_positions = runtime_metadata.query_positions.to(device=block_starts.device, dtype=torch.long)
    if cu_q.numel() != cu_global.numel():
        raise ValueError(
            "cu_seqlens_q and global cu_seqlens must describe the same samples "
            "before building SMLA compressed-KV prefixes."
        )

    full_block_starts = block_starts.to(dtype=torch.long)
    batch_count = cu_q.numel() - 1
    q_lengths = torch.diff(cu_q)
    query_sample_indices = torch.repeat_interleave(
        torch.arange(batch_count, dtype=torch.long, device=full_block_starts.device),
        q_lengths,
    )
    if query_sample_indices.numel() != query_positions.numel():
        raise ValueError("cu_seqlens_q must match the flattened query_positions length.")

    is_identity_prefix = bool(is_identity_compact_order) and batch_count == 1
    sample_starts = cu_global[:-1]
    nonempty_samples = q_lengths > 0
    prefix_ori_lengths = torch.zeros_like(q_lengths)
    last_query_indices = (cu_q[1:] - 1)[nonempty_samples]
    prefix_ori_lengths[nonempty_samples] = (
        query_positions.index_select(0, last_query_indices) - sample_starts[nonempty_samples] + 1
    )
    prefix_cmp_lengths = torch.div(
        prefix_ori_lengths,
        compression_ratio,
        rounding_mode="floor",
    )
    prefix_residuals = torch.remainder(prefix_ori_lengths, compression_ratio)

    block_count = full_block_starts.numel()
    if is_identity_prefix:
        selected_count = int(prefix_cmp_lengths[0].item())
        if selected_count > block_count:
            raise ValueError(
                "identity compressed KV does not contain the prefix required by local queries: "
                f"expected_blocks={selected_count}, actual_blocks={block_count}."
            )
        selected_global_indices = torch.arange(
            selected_count,
            dtype=torch.long,
            device=full_block_starts.device,
        )
        selected_block_starts = full_block_starts.narrow(0, 0, selected_count)
        global_to_local = torch.full(
            (block_count,),
            -1,
            dtype=torch.long,
            device=full_block_starts.device,
        )
        if selected_count > 0:
            global_to_local[:selected_count] = selected_global_indices
        prefix_cmp_lengths = prefix_cmp_lengths.to(dtype=torch.int32)
        cu_seqlens_cmp_kv = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=selected_block_starts.device),
                torch.cumsum(prefix_cmp_lengths, dim=0, dtype=torch.int32),
            )
        )
        return DeepSeekV4CPAlignmentDescriptor(
            full_block_starts=full_block_starts,
            selected_global_indices=selected_global_indices,
            global_to_local=global_to_local,
            block_starts=selected_block_starts,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            cmp_residual_kv=prefix_residuals.to(dtype=torch.int32),
            is_identity_prefix=True,
        )

    global_to_local = torch.full(
        (block_count,),
        -1,
        dtype=torch.long,
        device=full_block_starts.device,
    )
    block_sample_indices = torch.searchsorted(cu_global, full_block_starts, right=True) - 1
    valid_block_samples = (block_sample_indices >= 0) & (block_sample_indices < batch_count)
    safe_block_sample_indices = block_sample_indices.clamp(min=0, max=batch_count - 1)
    block_offsets = full_block_starts - sample_starts.index_select(0, safe_block_sample_indices)
    block_local_indices = torch.div(block_offsets, compression_ratio, rounding_mode="floor")
    selected_mask = (
        valid_block_samples
        & (block_offsets >= 0)
        & (torch.remainder(block_offsets, compression_ratio) == 0)
        & (block_local_indices < prefix_cmp_lengths.index_select(0, safe_block_sample_indices))
    )
    selected_global_indices = torch.nonzero(selected_mask, as_tuple=False).reshape(-1)

    expected_sample_indices = torch.repeat_interleave(
        torch.arange(batch_count, dtype=torch.long, device=full_block_starts.device),
        prefix_cmp_lengths,
    )
    prefix_cmp_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=full_block_starts.device),
            torch.cumsum(prefix_cmp_lengths, dim=0),
        )
    )
    expected_local_indices = torch.arange(
        expected_sample_indices.numel(), dtype=torch.long, device=full_block_starts.device
    ) - prefix_cmp_offsets[:-1].index_select(0, expected_sample_indices)
    expected_block_starts = (
        sample_starts.index_select(0, expected_sample_indices) + expected_local_indices * compression_ratio
    )
    selected_block_starts = full_block_starts.index_select(0, selected_global_indices)
    if selected_global_indices.numel() != expected_block_starts.numel() or not torch.equal(
        selected_block_starts, expected_block_starts
    ):
        raise ValueError(
            "compressed KV does not contain the complete prefix required by local queries: "
            f"expected_blocks={expected_block_starts.numel()}, "
            f"actual_blocks={selected_global_indices.numel()}."
        )

    global_to_local[selected_global_indices] = expected_local_indices
    prefix_cmp_lengths = prefix_cmp_lengths.to(dtype=torch.int32)
    cu_seqlens_cmp_kv = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=selected_block_starts.device),
            torch.cumsum(prefix_cmp_lengths, dim=0, dtype=torch.int32),
        )
    )
    cmp_residual_kv = prefix_residuals.to(dtype=torch.int32)
    return DeepSeekV4CPAlignmentDescriptor(
        full_block_starts=full_block_starts,
        selected_global_indices=selected_global_indices,
        global_to_local=global_to_local,
        block_starts=selected_block_starts,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        is_identity_prefix=False,
    )


def align_deepseek_v4_cp_tensor(
    tensor: torch.Tensor,
    alignment: DeepSeekV4CPAlignmentDescriptor,
    seq_dim: int,
    tensor_name: str = "tensor",
) -> torch.Tensor:
    """Select operator-facing compressed-prefix rows from a global compact tensor."""
    if not torch.is_tensor(tensor):
        raise TypeError(f"{tensor_name} must be a torch.Tensor.")
    if not isinstance(alignment, DeepSeekV4CPAlignmentDescriptor):
        raise TypeError("alignment must be a DeepSeekV4CPAlignmentDescriptor.")
    if tensor.dim() == 0:
        raise ValueError(f"{tensor_name} must have a sequence dimension.")

    seq_dim = seq_dim % tensor.dim()
    expected_full_length = int(alignment.full_block_starts.numel())
    actual_full_length = int(tensor.shape[seq_dim])
    if actual_full_length != expected_full_length:
        raise ValueError(
            f"{tensor_name} sequence length must match global compact block count: "
            f"expected {expected_full_length}, got {actual_full_length}."
        )

    if alignment.is_identity_prefix:
        aligned = tensor.narrow(
            seq_dim,
            0,
            int(alignment.selected_global_indices.numel()),
        ).contiguous()
    else:
        selected = alignment.selected_global_indices.to(device=tensor.device)
        aligned = tensor.index_select(seq_dim, selected).contiguous()
    expected_aligned_length = int(alignment.selected_global_indices.numel())
    actual_aligned_length = int(aligned.shape[seq_dim])
    if actual_aligned_length != expected_aligned_length:
        raise ValueError(
            f"{tensor_name} sequence length must match operator-facing cu_seqlens_cmp_kv: "
            f"expected {expected_aligned_length}, got {actual_aligned_length}."
        )
    return aligned


def remap_deepseek_v4_cp_sparse_indices(
    cmp_sparse_indices: torch.Tensor,
    alignment: DeepSeekV4CPAlignmentDescriptor,
) -> torch.Tensor:
    """Convert global compact-block indices to per-sample SMLA prefix indices."""
    if alignment.is_identity_prefix:
        return cmp_sparse_indices
    non_padding = cmp_sparse_indices >= 0
    if not torch.any(non_padding):
        return cmp_sparse_indices
    global_to_local = alignment.global_to_local
    if global_to_local.numel() == 0:
        raise ValueError("cmp_sparse_indices selects compressed blocks, but the local SMLA prefix is empty.")

    lookup = global_to_local.to(device=cmp_sparse_indices.device)
    flat_indices = cmp_sparse_indices.clamp(min=0).reshape(-1).to(dtype=torch.long)
    if torch.any(flat_indices >= lookup.numel()):
        value = int(flat_indices[flat_indices >= lookup.numel()][0].item())
        raise ValueError(
            "cmp_sparse_indices selects a block outside the global compact block range: "
            f"compact_index={value}, block_count={lookup.numel()}."
        )
    gathered = lookup.index_select(0, flat_indices)
    gathered = gathered.reshape(cmp_sparse_indices.shape)
    missing = non_padding & (gathered < 0)
    if torch.any(missing):
        value = int(cmp_sparse_indices[missing][0].item())
        raise ValueError(
            "cmp_sparse_indices selects a block outside the compressed-KV prefix required by local queries: "
            f"compact_index={value}."
        )
    return torch.where(non_padding, gathered.to(dtype=torch.int32), cmp_sparse_indices.new_full((), -1))


def _fill_deepseek_v4_smla_cmp_sparse_fallback(
    cmp_sparse_indices: torch.Tensor,
    query_positions: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_cmp_kv: torch.Tensor,
    cu_seqlens_ori_kv_global: torch.Tensor,
    compression_ratio: int,
) -> torch.Tensor:
    """Give all-padding rows the first causally visible block in their sample."""
    q_lengths = torch.diff(cu_seqlens_q.to(device=cmp_sparse_indices.device, dtype=torch.long))
    cmp_lengths = torch.diff(cu_seqlens_cmp_kv.to(device=cmp_sparse_indices.device, dtype=torch.long))
    cu_global = cu_seqlens_ori_kv_global.to(device=cmp_sparse_indices.device, dtype=torch.long)
    query_positions = query_positions.to(device=cmp_sparse_indices.device, dtype=torch.long).reshape(-1)
    if q_lengths.numel() != cmp_lengths.numel() or cu_global.numel() != q_lengths.numel() + 1:
        raise ValueError("Q, compressed-KV, and global cu_seqlens must describe the same SMLA samples.")
    if query_positions.numel() != cmp_sparse_indices.shape[0]:
        raise ValueError("query_positions must match the flattened cmp_sparse_indices query dimension.")

    sample_indices = torch.repeat_interleave(
        torch.arange(q_lengths.numel(), dtype=torch.long, device=cmp_sparse_indices.device),
        q_lengths,
    )

    sample_starts = cu_global[:-1].index_select(0, sample_indices)
    first_block_ends = sample_starts + compression_ratio
    fallback_rows = (cmp_sparse_indices < 0).all(dim=-1)
    fallback_rows &= cmp_lengths.index_select(0, sample_indices) > 0
    fallback_rows &= first_block_ends <= query_positions + 1
    if not torch.any(fallback_rows):
        return cmp_sparse_indices

    cmp_sparse_indices = cmp_sparse_indices.clone()
    cmp_sparse_indices[fallback_rows, 0] = 0
    return cmp_sparse_indices


def _get_deepseek_v4_cmp_visibility_inputs(
    runtime_metadata: DeepSeekV4CPRuntimeMetadata,
    device,
):
    if runtime_metadata.layout_q != "BSND":
        return (
            runtime_metadata.query_positions,
            runtime_metadata.cu_seqlens_ori_kv_global,
        )
    local_query_positions, _, global_seq_len = _get_deepseek_v4_bsnd_shared_query_metadata(
        runtime_metadata,
        device,
    )
    shared_cu_seqlens = torch.tensor(
        [0, global_seq_len],
        dtype=torch.int32,
        device=device,
    )
    return local_query_positions, shared_cu_seqlens


def build_deepseek_v4_cp_smla_inputs(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    compression_ratio: int,
    runtime_metadata: DeepSeekV4CPRuntimeMetadata,
    prepared_compressed_kv: Optional[DeepSeekV4CPCompressedKV] = None,
    cmp_sparse_indices: Optional[torch.Tensor] = None,
    cmp_sparse_indices_are_causal: bool = False,
    layout_q: str = "TND",
    layout_kv: str = "TND",
    metadata: Optional[torch.Tensor] = None,
    compacted_compressed_kv: Optional[torch.Tensor] = None,
    compacted_block_starts: Optional[torch.Tensor] = None,
) -> DeepSeekV4CPSMLAInputs:
    """Build the SMLA-facing input bundle for a DeepSeek V4 CP shard.

    ``runtime_metadata`` must come from ``build_deepseek_v4_owned_runtime_metadata``.
    C4A requires ``cmp_sparse_indices`` from the caller.
    """
    if runtime_metadata is None:
        raise ValueError("runtime_metadata is required.")
    if runtime_metadata.layout_q != layout_q or runtime_metadata.layout_kv != layout_kv:
        raise ValueError(
            "runtime_metadata layouts must match layout_q and layout_kv: "
            f"metadata=({runtime_metadata.layout_q}, {runtime_metadata.layout_kv}), "
            f"arguments=({layout_q}, {layout_kv})."
        )
    if runtime_metadata.batch_size != _get_deepseek_v4_batch_size(q, layout_q):
        raise ValueError("runtime_metadata batch_size must match the current q tensor.")
    if compression_ratio > 1 and prepared_compressed_kv is None:
        raise ValueError("prepared_compressed_kv is required when compression_ratio > 1.")
    if compression_ratio == 4 and cmp_sparse_indices is None:
        raise ValueError("cmp_sparse_indices is required for C4A.")

    cmp_kv = None
    smla_cmp_sparse_indices = None
    block_starts = None
    alignment = None
    cu_seqlens_cmp_kv = runtime_metadata.cu_seqlens_cmp_kv
    cmp_residual_kv = runtime_metadata.cmp_residual_kv
    seqused_cmp_kv = runtime_metadata.seqused_cmp_kv

    if cmp_sparse_indices is not None and compression_ratio != 4:
        raise ValueError("cmp_sparse_indices direct pass is only supported for C4A compression_ratio=4.")

    if compression_ratio > 1:
        if compacted_compressed_kv is not None:
            if compacted_block_starts is None:
                raise ValueError("compacted_block_starts is required when compacted_compressed_kv is provided.")
            full_cmp_kv = compacted_compressed_kv
            full_block_starts = compacted_block_starts
        else:
            full_cmp_kv, full_block_starts = compact_deepseek_v4_compressed_kv(
                prepared_compressed_kv,
                seq_dim=runtime_metadata.cmp_seq_dim,
            )
        alignment = _build_deepseek_v4_cp_alignment_descriptor(
            full_block_starts,
            runtime_metadata,
            compression_ratio,
            is_identity_compact_order=bool(prepared_compressed_kv.metadata.is_identity_compact_order),
        )
        cmp_kv = align_deepseek_v4_cp_tensor(
            full_cmp_kv,
            alignment,
            runtime_metadata.cmp_seq_dim,
            tensor_name="cmp_kv",
        )
        block_starts = alignment.block_starts
        cu_seqlens_cmp_kv = alignment.cu_seqlens_cmp_kv
        cmp_residual_kv = alignment.cmp_residual_kv
        if metadata is not None:
            prefix_changed = not torch.equal(
                cu_seqlens_cmp_kv,
                runtime_metadata.cu_seqlens_cmp_kv.to(device=cu_seqlens_cmp_kv.device),
            ) or not torch.equal(
                cmp_residual_kv,
                runtime_metadata.cmp_residual_kv.to(device=cmp_residual_kv.device),
            )
            if prefix_changed:
                raise ValueError(
                    "Precomputed SMLA metadata is invalid after compressed-KV prefix cropping; "
                    "pass metadata=None so it is rebuilt from the operator-facing prefix."
                )
        if compression_ratio == 4:
            valid_mask = torch.ones_like(full_block_starts, dtype=torch.bool)
            visibility_query_positions, visibility_cu_seqlens = _get_deepseek_v4_cmp_visibility_inputs(
                runtime_metadata,
                full_block_starts.device,
            )
            flat_sparse_indices = _flatten_deepseek_v4_cmp_sparse_indices(
                cmp_sparse_indices,
                q,
                layout_q,
            )
            if not (cmp_sparse_indices_are_causal and alignment.is_identity_prefix):
                flat_sparse_indices = validate_deepseek_v4_c4a_cmp_sparse_indices(
                    flat_sparse_indices,
                    visibility_query_positions,
                    full_block_starts,
                    valid_mask,
                    compression_ratio,
                    cu_seqlens=visibility_cu_seqlens,
                )
            flat_sparse_indices = remap_deepseek_v4_cp_sparse_indices(
                flat_sparse_indices,
                alignment,
            )
            flat_sparse_indices = _fill_deepseek_v4_smla_cmp_sparse_fallback(
                flat_sparse_indices,
                runtime_metadata.query_positions,
                runtime_metadata.cu_seqlens_q,
                cu_seqlens_cmp_kv,
                runtime_metadata.cu_seqlens_ori_kv_global,
                compression_ratio,
            )
            smla_cmp_sparse_indices = _reshape_deepseek_v4_sparse_indices_for_layout(
                flat_sparse_indices.contiguous(),
                q,
                layout_q,
            )

    return DeepSeekV4CPSMLAInputs(
        q=q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=smla_cmp_sparse_indices,
        cu_seqlens_q=runtime_metadata.cu_seqlens_q,
        cu_seqlens_ori_kv=runtime_metadata.cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_ori_kv=runtime_metadata.seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        metadata=metadata,
        block_starts=block_starts,
        alignment=alignment,
    )


def flatten_deepseek_v4_cp_tensor_to_tnd(
    tensor: torch.Tensor,
    seq_dim: int = 0,
    batch_dim: Optional[int] = 1,
    shared_kv: bool = False,
) -> torch.Tensor:
    """Flatten [S, B, ...] model tensors to batch-major TND tensors.

    Query tensors shaped [S, B, N, D] become [B*S, N, D]. Shared KV tensors
    shaped [S, B, D] become [B*S, 1, D] when ``shared_kv`` is True.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("tensor must be a torch.Tensor.")
    if tensor.dim() == 0:
        raise ValueError("tensor must have a sequence dimension.")
    if batch_dim is None:
        flattened = torch.movedim(tensor, seq_dim, 0).contiguous()
    else:
        dim_count = tensor.dim()
        seq_dim = seq_dim % dim_count
        batch_dim = batch_dim % dim_count
        if seq_dim == batch_dim:
            raise ValueError("seq_dim and batch_dim must be different.")
        flattened = torch.movedim(tensor, (batch_dim, seq_dim), (0, 1)).contiguous().flatten(0, 1)
    if shared_kv and flattened.dim() == 2:
        flattened = flattened.unsqueeze(1)
    return flattened.contiguous()
