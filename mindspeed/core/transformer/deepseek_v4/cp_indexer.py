# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""MindSpeed-owned DeepSeek V4 CP Indexer orchestration.

The model-owned :class:`DSAIndexer` computes query/key representations.  This
module owns the CP-specific parts around it: fused top-k execution, causal
segment alignment, and conversion from global compressed-block ordinals to the
compact SMLA key space.
"""

from functools import lru_cache

import torch

from mindspeed.core.context_parallel.deepseek_v4_context_parallel import (
    flatten_deepseek_v4_cp_tensor_to_tnd,
)
from mindspeed.core.context_parallel.deepseek_v4_context_parallel._utils import (
    enumerate_global_block_starts,
    normalize_cu_seqlens,
)


def _flatten_indexer_sparse_indices_to_tnd(indices, seq_len, batch_size):
    if indices is None or indices.dim() == 2:
        return indices
    if indices.dim() == 4 and indices.shape[2] == 1:
        indices = indices.squeeze(2)
    if indices.dim() != 3:
        raise ValueError("Lightning Indexer sparse indices must be 2-D, 3-D, or singleton-head 4-D.")
    if indices.shape[0] == batch_size and indices.shape[1] == seq_len:
        return indices.reshape(batch_size * seq_len, indices.shape[-1]).contiguous()
    if indices.shape[0] == seq_len and indices.shape[1] == batch_size:
        return indices.transpose(0, 1).reshape(batch_size * seq_len, indices.shape[-1]).contiguous()
    if indices.shape[1] == 1 and indices.shape[0] == batch_size * seq_len:
        return indices.squeeze(1).contiguous()
    raise ValueError("Lightning Indexer sparse indices do not match the local query sequence and batch dimensions.")


def _normalize_bsnd_indexer_sparse_indices(indices, seq_len, batch_size):
    if indices is None:
        return None
    if indices.dim() == 4 and indices.shape[2] == 1:
        indices = indices.squeeze(2)
    if indices.dim() == 2 and batch_size == 1 and indices.shape[0] == seq_len:
        return indices.unsqueeze(0).contiguous()
    if indices.dim() != 3:
        raise ValueError("BSND Lightning Indexer sparse indices must be 3-D or singleton-head 4-D.")
    if indices.shape[0] == batch_size and indices.shape[1] == seq_len:
        return indices.contiguous()
    if indices.shape[0] == seq_len and indices.shape[1] == batch_size:
        return indices.transpose(0, 1).contiguous()
    raise ValueError("BSND Lightning Indexer sparse indices do not match the local sequence and batch dimensions.")


@lru_cache(maxsize=64)
def _cached_global_compressed_token_starts(cu_values, compression_ratio, device):
    if int(compression_ratio) <= 0:
        raise ValueError("compression_ratio must be positive.")
    cu = normalize_cu_seqlens(cu_values, device).to(dtype=torch.long)
    return enumerate_global_block_starts(cu, int(compression_ratio))


def get_global_compressed_token_starts(cu_seqlens_global, compression_ratio, device):
    if torch.is_tensor(cu_seqlens_global):
        values = tuple(int(value) for value in cu_seqlens_global.detach().cpu().tolist())
    else:
        values = tuple(int(value) for value in cu_seqlens_global)
    return _cached_global_compressed_token_starts(values, int(compression_ratio), str(device)).to(device=device)


def map_global_cmp_indices_to_compact(
    global_indices,
    block_starts,
    compression_ratio,
    cu_seqlens_global,
):
    """Map global compressed-block ordinals to compact compressed-KV indices."""
    if global_indices is None:
        return None
    if not torch.is_tensor(block_starts) or block_starts.numel() == 0:
        return torch.full_like(global_indices, -1)

    device = global_indices.device
    block_starts = block_starts.to(device=device, dtype=torch.long).reshape(-1)
    valid_starts = block_starts[block_starts >= 0]
    if valid_starts.numel() == 0:
        return torch.full_like(global_indices, -1)

    lookup = torch.full(
        (int(valid_starts.max().item()) + 1,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    lookup[valid_starts] = torch.arange(valid_starts.numel(), dtype=torch.int32, device=device)

    flat_indices = global_indices.reshape(-1).to(dtype=torch.long)
    token_starts = torch.full_like(flat_indices, -1)
    ordinal_starts = get_global_compressed_token_starts(
        cu_seqlens_global,
        compression_ratio,
        device,
    )
    valid_ordinals = (flat_indices >= 0) & (flat_indices < ordinal_starts.numel())
    if torch.any(valid_ordinals):
        token_starts[valid_ordinals] = ordinal_starts.index_select(0, flat_indices[valid_ordinals])
    in_range = (token_starts >= 0) & (token_starts < lookup.numel())
    compact = torch.full_like(flat_indices, -1, dtype=torch.int32)
    compact[in_range] = lookup[token_starts[in_range]]
    return compact.reshape(global_indices.shape)


def filter_deepseek_v4_causal_compact_indices(
    compact_indices,
    query_positions,
    block_starts,
    compression_ratio,
    cu_seqlens_global,
):
    """Drop future and cross-sample blocks from compact Indexer output."""
    if compact_indices.dim() != 2:
        raise ValueError("flattened compact indices must have shape [query_count, topk].")
    if compact_indices.shape[0] != query_positions.numel():
        raise ValueError("compact indices query dimension must match query_positions.")

    device = compact_indices.device
    query_positions = query_positions.to(device=device, dtype=torch.long).reshape(-1)
    block_starts = block_starts.to(device=device, dtype=torch.long).reshape(-1)
    cu = normalize_cu_seqlens(cu_seqlens_global, device).to(dtype=torch.long)
    if block_starts.numel() == 0:
        return torch.full_like(compact_indices, -1)

    non_padding = compact_indices >= 0
    in_range = non_padding & (compact_indices < block_starts.numel())
    safe_indices = compact_indices.clamp(min=0, max=block_starts.numel() - 1).reshape(-1).long()
    starts = block_starts.index_select(0, safe_indices).reshape(compact_indices.shape)
    ends = starts + int(compression_ratio)
    query_sample = torch.searchsorted(cu, query_positions, right=True) - 1
    block_sample = torch.searchsorted(cu, starts, right=True) - 1
    block_end_sample = torch.searchsorted(cu, ends - 1, right=True) - 1
    sample_count = cu.numel() - 1
    visible = (
        in_range
        & (query_sample.view(-1, 1) >= 0)
        & (query_sample.view(-1, 1) < sample_count)
        & (block_sample >= 0)
        & (block_sample < sample_count)
        & (block_sample == block_end_sample)
        & (block_sample == query_sample.view(-1, 1))
        & (ends <= (query_positions + 1).view(-1, 1))
    )
    return torch.where(visible, compact_indices, compact_indices.new_full((), -1))


def _pad_topk(indices, topk):
    if indices.shape[-1] >= topk:
        return indices[..., :topk]
    return torch.cat(
        (
            indices,
            indices.new_full((*indices.shape[:-1], topk - indices.shape[-1]), -1),
        ),
        dim=-1,
    )


def _dense_index_tnd(
    query_index,
    key_index,
    weights,
    query_positions,
    block_starts,
    compression_ratio,
    cu_seqlens_global,
    topk,
    query_chunk_size,
    head_chunk_size,
):
    if query_index.dim() == 4:
        query = flatten_deepseek_v4_cp_tensor_to_tnd(query_index, seq_dim=0, batch_dim=1)
    elif query_index.dim() == 3:
        query = query_index.contiguous()
    else:
        raise ValueError("TND dense Indexer query must have shape [T, N, D] or [T, B, N, D].")
    if weights.dim() == 3:
        weight = flatten_deepseek_v4_cp_tensor_to_tnd(weights, seq_dim=0, batch_dim=1).float()
    elif weights.dim() == 2:
        weight = weights.contiguous().float()
    else:
        raise ValueError("TND dense Indexer weights must have shape [T, N] or [T, B, N].")
    if query.dim() != 3 or weight.dim() != 2 or key_index.dim() != 3 or key_index.shape[1] != 1:
        raise ValueError("TND dense Indexer expects query [T, N, D], key [K, 1, D], weights [T, N].")
    if query.shape[0] != weight.shape[0] or query.shape[1] != weight.shape[1]:
        raise ValueError("TND dense Indexer query and weights dimensions must match.")
    if query.shape[-1] != key_index.shape[-1]:
        raise ValueError("TND dense Indexer query and key dimensions must match.")

    device = query.device
    positions = query_positions.to(device=device, dtype=torch.long).reshape(-1)
    starts = block_starts.to(device=device, dtype=torch.long).reshape(-1)
    cu = normalize_cu_seqlens(cu_seqlens_global, device).to(dtype=torch.long)
    if positions.numel() != query.shape[0] or starts.numel() != key_index.shape[0]:
        raise ValueError("TND dense Indexer tensors do not match CP metadata.")
    result = torch.full((query.shape[0], int(topk)), -1, dtype=torch.int32, device=device)
    if starts.numel() == 0:
        return result

    key = key_index[:, 0, :].float()
    block_end = starts + int(compression_ratio)
    block_sample = torch.searchsorted(cu, starts, right=True) - 1
    block_end_sample = torch.searchsorted(cu, block_end - 1, right=True) - 1
    valid_blocks = (block_sample >= 0) & (block_sample < cu.numel() - 1) & (block_sample == block_end_sample)
    actual_topk = min(int(topk), key.shape[0])
    for begin in range(0, query.shape[0], int(query_chunk_size)):
        end = min(begin + int(query_chunk_size), query.shape[0])
        scores = torch.zeros((end - begin, key.shape[0]), dtype=torch.float32, device=device)
        for head_begin in range(0, query.shape[1], int(head_chunk_size)):
            head_end = min(head_begin + int(head_chunk_size), query.shape[1])
            logits = torch.matmul(
                query[begin:end, head_begin:head_end].float(),
                key.transpose(0, 1),
            )
            scores.add_((torch.relu(logits) * weight[begin:end, head_begin:head_end].unsqueeze(-1)).sum(dim=1))
        q_sample = torch.searchsorted(cu, positions[begin:end], right=True) - 1
        visible = (
            valid_blocks.view(1, -1)
            & (q_sample.view(-1, 1) >= 0)
            & (q_sample.view(-1, 1) < cu.numel() - 1)
            & (block_sample.view(1, -1) == q_sample.view(-1, 1))
            & (block_end.view(1, -1) <= (positions[begin:end] + 1).view(-1, 1))
        )
        scores.masked_fill_(~visible, float("-inf"))
        if actual_topk:
            values, indices = torch.topk(scores, actual_topk, dim=-1)
            indices = torch.where(torch.isfinite(values), indices, indices.new_full((), -1))
            result[begin:end, :actual_topk] = indices.to(torch.int32)
    return result


def _dense_index_bsnd(
    query_index,
    key_index,
    weights,
    query_positions,
    block_starts,
    compression_ratio,
    cu_seqlens_global,
    topk,
    query_chunk_size,
    head_chunk_size,
):
    if query_index.dim() != 4 or weights.dim() != 3:
        raise ValueError("BSND dense Indexer expects model query [S, B, N, D] and weights [S, B, N].")
    seq_len, batch_size, head_count, head_dim = query_index.shape
    if tuple(weights.shape) != (seq_len, batch_size, head_count):
        raise ValueError("BSND dense Indexer query and weights dimensions must match.")
    if key_index.dim() != 4 or key_index.shape[0] != batch_size or key_index.shape[2] != 1:
        raise ValueError("BSND dense Indexer key must have shape [B, K, 1, D].")
    if key_index.shape[-1] != head_dim:
        raise ValueError("BSND dense Indexer query and key dimensions must match.")

    device = query_index.device
    positions = query_positions.to(device=device, dtype=torch.long).reshape(batch_size, seq_len)
    cu = normalize_cu_seqlens(cu_seqlens_global, device).to(dtype=torch.long)
    if cu.numel() != batch_size + 1:
        raise ValueError("BSND dense Indexer global cu_seqlens must contain one boundary per batch.")
    starts = block_starts.to(device=device, dtype=torch.long).reshape(-1)
    if starts.numel() != key_index.shape[1]:
        raise ValueError("BSND dense Indexer block_starts must match key length.")
    local_positions = positions - cu[:-1].unsqueeze(1)
    if not torch.equal(local_positions, local_positions[:1].expand_as(local_positions)):
        raise ValueError("BSND dense Indexer batches must share one query position plan.")

    result = torch.full((batch_size, seq_len, int(topk)), -1, dtype=torch.int32, device=device)
    if starts.numel() == 0:
        return result
    key = key_index[:, :, 0, :].float()
    block_end = starts + int(compression_ratio)
    valid_blocks = (starts >= 0) & (block_end <= local_positions.max().item() + int(compression_ratio))
    actual_topk = min(int(topk), starts.numel())
    key_t = key.transpose(-1, -2).unsqueeze(1)
    for begin in range(0, seq_len, int(query_chunk_size)):
        end = min(begin + int(query_chunk_size), seq_len)
        scores = torch.zeros((batch_size, end - begin, starts.numel()), dtype=torch.float32, device=device)
        for head_begin in range(0, head_count, int(head_chunk_size)):
            head_end = min(head_begin + int(head_chunk_size), head_count)
            logits = torch.matmul(
                query_index[begin:end].transpose(0, 1).contiguous()[:, :, head_begin:head_end].float(),
                key_t,
            )
            scores.add_(
                (
                    torch.relu(logits)
                    * weights[begin:end].transpose(0, 1).contiguous()[:, :, head_begin:head_end].float().unsqueeze(-1)
                ).sum(dim=2)
            )
        visible = valid_blocks.view(1, 1, -1) & (
            block_end.view(1, 1, -1) <= (local_positions[:, begin:end] + 1).unsqueeze(-1)
        )
        scores.masked_fill_(~visible, float("-inf"))
        if actual_topk:
            values, indices = torch.topk(scores, actual_topk, dim=-1)
            indices = torch.where(torch.isfinite(values), indices, indices.new_full((), -1))
            result[:, begin:end, :actual_topk] = indices.to(torch.int32)
    return result


def build_deepseek_v4_dense_indexer_compact_indices(
    query_index,
    key_index,
    weights,
    query_positions,
    block_starts,
    compression_ratio,
    cu_seqlens_global,
    topk,
    query_chunk_size=64,
    head_chunk_size=8,
    layout="TND",
):
    if int(topk) <= 0:
        raise ValueError("Indexer topk must be positive.")
    if int(query_chunk_size) <= 0 or int(head_chunk_size) <= 0:
        raise ValueError("Indexer query_chunk_size and head_chunk_size must be positive.")
    if layout == "TND":
        return _dense_index_tnd(
            query_index,
            key_index,
            weights,
            query_positions,
            block_starts,
            compression_ratio,
            cu_seqlens_global,
            topk,
            query_chunk_size,
            head_chunk_size,
        )
    if layout == "BSND":
        return _dense_index_bsnd(
            query_index,
            key_index,
            weights,
            query_positions,
            block_starts,
            compression_ratio,
            cu_seqlens_global,
            topk,
            query_chunk_size,
            head_chunk_size,
        )
    raise ValueError(f"Unsupported dense Indexer layout: {layout}.")


def _tnd_fused_indexer_segment_bounds(
    positions,
    sample_ids,
    identity_single_sample,
):
    query_count = int(positions.numel())
    device = positions.device
    if identity_single_sample:
        return torch.tensor([0, query_count], dtype=torch.long, device=device)
    new_segment = torch.ones(query_count, dtype=torch.bool, device=device)
    if query_count > 1:
        new_segment[1:] = (sample_ids[1:] != sample_ids[:-1]) | (positions[1:] != positions[:-1] + 1)
    starts = torch.nonzero(new_segment, as_tuple=False).flatten()
    return torch.cat((starts, starts.new_full((1,), query_count)))


def _normalize_fused_output(indices, values, layout, seq_len, batch_size):
    if layout == "BSND":
        indices = _normalize_bsnd_indexer_sparse_indices(indices, seq_len, batch_size)
        if values is not None:
            values = _normalize_bsnd_indexer_sparse_indices(values, seq_len, batch_size)
        return indices, values
    indices = _flatten_indexer_sparse_indices_to_tnd(indices, seq_len, batch_size)
    if values is not None:
        values = _flatten_indexer_sparse_indices_to_tnd(values, seq_len, batch_size)
    return indices, values


def run_deepseek_v4_right_aligned_fused_indexer(
    indexer,
    dsa_hidden_states,
    query_index,
    key_index,
    weights,
    query_positions,
    cu_seqlens_global,
    compression_ratio,
    topk,
    start_pos,
    identity_single_sample=False,
    layout="TND",
):
    """Run fused Lightning Indexer on causal right-aligned query segments."""
    if layout == "BSND":
        return _run_deepseek_v4_bsnd_right_aligned_fused_indexer(
            indexer,
            dsa_hidden_states,
            query_index,
            key_index,
            weights,
            query_positions,
            cu_seqlens_global,
            compression_ratio,
            topk,
            start_pos,
        )
    if layout != "TND":
        raise ValueError(f"Unsupported fused Indexer layout: {layout}.")
    ratio = int(compression_ratio)
    topk = int(topk)
    if ratio <= 0 or topk <= 0:
        raise ValueError("fused Indexer compression_ratio and topk must be positive.")

    query = flatten_deepseek_v4_cp_tensor_to_tnd(query_index, seq_dim=0, batch_dim=1)
    weight = flatten_deepseek_v4_cp_tensor_to_tnd(weights, seq_dim=0, batch_dim=1)
    hidden = flatten_deepseek_v4_cp_tensor_to_tnd(dsa_hidden_states, seq_dim=0, batch_dim=1)
    positions = query_positions.to(device=query.device, dtype=torch.long).reshape(-1)
    cu = normalize_cu_seqlens(cu_seqlens_global, query.device, trusted=True).to(dtype=torch.long)
    if query.dim() != 3 or weight.dim() != 2 or hidden.dim() != 2:
        raise ValueError("TND fused Indexer expects query [T, N, D], weights [T, N], hidden [T, D].")
    if positions.numel() != query.shape[0] or hidden.shape[0] != query.shape[0]:
        raise ValueError("TND fused Indexer inputs do not match query positions.")
    if positions.numel() == 0:
        return torch.empty((0, topk), dtype=torch.int32, device=query.device)
    sample_count = cu.numel() - 1
    identity_single_sample = bool(identity_single_sample) and sample_count == 1
    if identity_single_sample:
        sample_ids = torch.zeros_like(positions)
    else:
        sample_ids = torch.searchsorted(cu, positions, right=True) - 1
    if torch.any(sample_ids < 0) or torch.any(sample_ids >= sample_count):
        raise ValueError("TND fused Indexer query positions must stay inside sample boundaries.")
    if identity_single_sample:
        block_starts = (
            torch.arange(
                int(cu[-1].item()) // ratio,
                dtype=torch.long,
                device=query.device,
            )
            * ratio
        )
    else:
        block_starts = get_global_compressed_token_starts(cu, ratio, query.device)
    segment_bounds = _tnd_fused_indexer_segment_bounds(
        positions,
        sample_ids,
        identity_single_sample,
    )
    segment_starts = segment_bounds[:-1]
    segment_ends = segment_bounds[1:]
    sample_ids_seg = sample_ids.index_select(0, segment_starts)
    query_ends = positions.index_select(0, segment_ends - 1) + 1
    sample_starts_t = cu.index_select(0, sample_ids_seg)
    sample_limits_t = cu.index_select(0, sample_ids_seg + 1)
    if identity_single_sample:
        block_begins = torch.zeros_like(segment_starts)
        block_ends = torch.full_like(segment_starts, block_starts.numel())
    else:
        block_begins = torch.searchsorted(block_starts, sample_starts_t, right=False)
        block_ends = torch.searchsorted(block_starts, sample_limits_t, right=False)
    visible_counts = torch.div(query_ends - sample_starts_t, ratio, rounding_mode="floor")
    residuals = (query_ends - sample_starts_t) - visible_counts * ratio
    plan = torch.stack(
        (
            segment_starts.to(dtype=torch.long),
            segment_ends.to(dtype=torch.long),
            sample_starts_t.to(dtype=torch.long),
            query_ends.to(dtype=torch.long),
            block_begins.to(dtype=torch.long),
            block_ends.to(dtype=torch.long),
            visible_counts.to(dtype=torch.long),
            residuals.to(dtype=torch.long),
        ),
        dim=1,
    )
    result_chunks = []
    for (
        segment_start,
        segment_end,
        sample_start,
        query_end,
        block_begin,
        block_end,
        visible_count,
        residual_val,
    ) in plan.tolist():
        key_end = block_begin + visible_count
        q_segment = query[segment_start:segment_end].unsqueeze(1).contiguous()
        w_segment = weight[segment_start:segment_end].unsqueeze(1).contiguous()
        if visible_count <= 0:
            indices = torch.full((segment_end - segment_start, topk), -1, dtype=torch.int32, device=query.device)
        else:
            if key_end > block_end:
                raise ValueError("fused Indexer causal prefix exceeds the sample block range.")
            key_segment = key_index[block_begin:key_end].unsqueeze(1).contiguous()
            residual = torch.tensor(
                [residual_val],
                dtype=torch.int32,
                device=query.device,
            )
            indices, _ = indexer.forward_with_scores_compress(
                q_segment,
                key_segment,
                w_segment,
                None,
                topk,
                block_begin,
                ratio,
                cmp_residual_k=residual,
                return_scores=False,
            )
            indices, _ = _normalize_fused_output(
                indices,
                None,
                "BSND",
                segment_end - segment_start,
                1,
            )
            indices = indices.squeeze(0)
        result_chunks.append(indices)
    return (
        torch.cat(result_chunks, dim=0).contiguous()
        if result_chunks
        else torch.empty((0, topk), dtype=torch.int32, device=query.device)
    )


def _run_deepseek_v4_bsnd_right_aligned_fused_indexer(
    indexer,
    dsa_hidden_states,
    query_index,
    key_index,
    weights,
    query_positions,
    cu_seqlens_global,
    compression_ratio,
    topk,
    start_pos,
):
    if query_index.dim() != 4 or weights.dim() != 3 or dsa_hidden_states.dim() != 3:
        raise ValueError("BSND fused Indexer expects model-layout query, weights, and hidden states.")
    seq_len, batch_size, _, _ = query_index.shape
    if tuple(weights.shape[:2]) != (seq_len, batch_size):
        raise ValueError("BSND fused Indexer query and weights dimensions must match.")
    if tuple(dsa_hidden_states.shape[:2]) != (seq_len, batch_size):
        raise ValueError("BSND fused Indexer query and hidden-state dimensions must match.")
    if key_index.dim() != 4 or key_index.shape[0] != batch_size or key_index.shape[2] != 1:
        raise ValueError("BSND fused Indexer key must have shape [B,K,1,D].")

    ratio = int(compression_ratio)
    topk = int(topk)
    if ratio <= 0 or topk <= 0:
        raise ValueError("BSND fused Indexer compression_ratio and TopK must be positive.")
    device = query_index.device
    cu_global = normalize_cu_seqlens(cu_seqlens_global, device, trusted=True).to(dtype=torch.long)
    if cu_global.numel() != batch_size + 1:
        raise ValueError("BSND fused Indexer cu_seqlens must contain one boundary per batch.")
    global_lengths = torch.diff(cu_global)
    if global_lengths.numel() == 0 or not bool((global_lengths == global_lengths[0]).all().item()):
        raise ValueError("BSND fused Indexer requires equal global sequence lengths.")
    global_seq_len = int(global_lengths[0].item())
    expected_key_count = global_seq_len // ratio
    if key_index.shape[1] != expected_key_count:
        raise ValueError(
            "BSND fused Indexer key length must match the shared compressed sequence: "
            f"key_count={key_index.shape[1]}, expected={expected_key_count}."
        )

    positions = query_positions.to(device=device, dtype=torch.long).reshape(-1)
    if positions.numel() == seq_len:
        shared_positions = positions
    elif positions.numel() == batch_size * seq_len:
        position_matrix = positions.reshape(batch_size, seq_len)
        local_positions = position_matrix - cu_global[:-1].unsqueeze(1)
        if not torch.equal(local_positions, local_positions[:1].expand_as(local_positions)):
            raise ValueError("BSND fused Indexer batches must share one query position plan.")
        shared_positions = local_positions[0]
    else:
        raise ValueError("BSND fused Indexer query_positions must be shared or batch-flattened.")
    if shared_positions.numel() == 0:
        return torch.empty((batch_size, 0, topk), dtype=torch.int32, device=device)
    if torch.any(shared_positions < 0) or torch.any(shared_positions >= global_seq_len):
        raise ValueError("BSND fused Indexer query positions must stay inside the shared sequence.")

    n_pos = int(shared_positions.numel())
    new_segment = torch.ones(n_pos, dtype=torch.bool, device=device)
    if n_pos > 1:
        new_segment[1:] = shared_positions[1:] != shared_positions[:-1] + 1
    starts = torch.nonzero(new_segment, as_tuple=False).flatten()
    segment_bounds = torch.cat((starts, starts.new_full((1,), n_pos)))

    segment_starts = segment_bounds[:-1]
    segment_ends = segment_bounds[1:]
    original_k_lengths = shared_positions.index_select(0, segment_ends - 1) + 1
    visible_block_counts = torch.div(original_k_lengths, ratio, rounding_mode="floor")
    query_lengths = segment_ends - segment_starts
    residuals = torch.remainder(original_k_lengths, ratio)
    plan = torch.stack(
        (
            segment_starts.to(dtype=torch.long),
            segment_ends.to(dtype=torch.long),
            original_k_lengths.to(dtype=torch.long),
            visible_block_counts.to(dtype=torch.long),
            query_lengths.to(dtype=torch.long),
            residuals.to(dtype=torch.long),
        ),
        dim=1,
    )
    index_chunks = []
    for (
        segment_start,
        segment_end,
        original_k_length,
        visible_block_count,
        query_length,
        residual,
    ) in plan.tolist():
        if visible_block_count == 0:
            index_chunks.append(
                torch.full(
                    (batch_size, query_length, topk),
                    -1,
                    dtype=torch.int32,
                    device=device,
                )
            )
            continue

        query_segment = query_index[segment_start:segment_end].contiguous()
        weight_segment = weights[segment_start:segment_end].contiguous()
        key_segment = key_index[:, :visible_block_count].transpose(0, 1).contiguous()
        cmp_residual_k = torch.full(
            (batch_size,),
            residual,
            dtype=torch.int32,
            device=device,
        )
        indices, _ = indexer.forward_with_scores_compress(
            query_segment,
            key_segment,
            weight_segment,
            None,
            topk,
            0,
            ratio,
            cmp_residual_k=cmp_residual_k,
            return_scores=False,
        )
        indices, _ = _normalize_fused_output(indices, None, "BSND", query_length, batch_size)
        index_chunks.append(indices)
    return torch.cat(index_chunks, dim=1).contiguous()


def finalize_deepseek_v4_fused_indexer_compact_indices(
    global_indices,
    block_starts,
    compression_ratio,
    cu_seqlens_global,
    query_positions,
    identity_compact_order,
    layout,
    batch_size,
    seq_len,
):
    """Convert fused Indexer ordinals into compact SMLA indices.

    TND keeps the post-refactor path: always map, never filter, causal=True.
    BSND restores the pre-refactor contract: identity skips map/filter;
    non-identity maps then applies the Python causal filter.
    """
    if layout == "TND":
        compact = map_global_cmp_indices_to_compact(
            global_indices,
            block_starts,
            compression_ratio,
            cu_seqlens_global,
        )
        return compact, True
    if layout != "BSND":
        raise ValueError(f"Unsupported fused Indexer layout: {layout}.")

    indices = global_indices.to(dtype=torch.int32)
    if identity_compact_order:
        compact = _normalize_bsnd_indexer_sparse_indices(indices, seq_len, batch_size)
        return compact, True

    compact = map_global_cmp_indices_to_compact(
        indices,
        block_starts,
        compression_ratio,
        cu_seqlens_global,
    )
    compact = _normalize_bsnd_indexer_sparse_indices(compact, seq_len, batch_size)
    sparse_shape = compact.shape
    flat_indices = compact.reshape(int(batch_size) * int(seq_len), sparse_shape[-1])
    shared_positions = query_positions.to(device=compact.device, dtype=torch.long).reshape(-1)
    if shared_positions.numel() == int(seq_len):
        filter_positions = shared_positions.repeat(int(batch_size))
    elif shared_positions.numel() == int(batch_size) * int(seq_len):
        filter_positions = shared_positions
    else:
        raise ValueError("BSND fused Indexer query_positions must be shared or batch-flattened.")
    compact = filter_deepseek_v4_causal_compact_indices(
        flat_indices,
        filter_positions,
        block_starts,
        compression_ratio,
        cu_seqlens_global,
    ).reshape(sparse_shape)
    return compact, False


def should_use_deepseek_v4_cp_indexer_loss(attention, args, loss_coeff):
    configured = float(getattr(args, "indexer_loss_coeff", loss_coeff))
    return (
        float(loss_coeff) > 0
        and configured > 0
        and bool(getattr(args, "use_fused_lightning_indexer_loss", False))
        and bool(getattr(attention, "training", True))
        and torch.is_grad_enabled()
    )


__all__ = [
    "_flatten_indexer_sparse_indices_to_tnd",
    "_normalize_bsnd_indexer_sparse_indices",
    "build_deepseek_v4_dense_indexer_compact_indices",
    "filter_deepseek_v4_causal_compact_indices",
    "finalize_deepseek_v4_fused_indexer_compact_indices",
    "get_global_compressed_token_starts",
    "map_global_cmp_indices_to_compact",
    "run_deepseek_v4_right_aligned_fused_indexer",
    "should_use_deepseek_v4_cp_indexer_loss",
]
