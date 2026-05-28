# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from functools import wraps

import torch
import torch_npu
from einops import rearrange

from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from mindspeed.te.pytorch.attention.dot_product_attention.kvallgather_context_parallel import (
    get_seq_chunk_ids_for_reordering_before_attn,
    get_cu_seqlens_qkv_before_attn
)
from mindspeed.te.pytorch.attention.dot_product_attention.utils import (
    get_distributed_rank,
    get_distributed_world_size
)


def gather_and_permute_cp_shard(
    t: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup
    ) -> torch.Tensor:

    cp_size = get_distributed_world_size(cp_group)

    # [s, ...] -> [cp, s, ...]
    t_ag = gather_from_sequence_parallel_region(t, group=cp_group)

    # [cp, s, ...] -> [cp*2, s//2, ...]
    t_ag = t_ag.view(2 * cp_size, t.shape[0] // 2, *t.shape[1:])

    chunk_ids = get_seq_chunk_ids_for_reordering_before_attn(cp_size, t.device)
    t_ag = torch.index_select(t_ag, dim=0, index=chunk_ids)

    # [cp*2, s//2, ...] -> [cp*s, ...]
    return t_ag.view(-1, *t.shape[1:])


def compute_cu_seqlens_for_window(cu_seqlens, window_size):
    """Truncate cu_seqlens to cover only tokens within [0, window_size).

    The reordered K follows global causal order, so we find which documents
    fit within the window and truncate the last one at the window boundary.

    Supports both formats:
      - With leading 0: tensor([0, 1024, 2048])
      - Without leading 0: tensor([1024, 2048])
    """
    # Normalize: ensure leading 0 for consistent iteration
    has_leading_zero = cu_seqlens.numel() > 0 and cu_seqlens[0].item() == 0
    if has_leading_zero:
        boundaries = cu_seqlens[1:]
    else:
        boundaries = cu_seqlens

    result = [0]
    for cs in boundaries:
        cs_val = cs.item() if isinstance(cs, torch.Tensor) else cs
        if cs_val >= window_size:
            result.append(window_size)
            break
        result.append(cs_val)
    return torch.tensor(result, dtype=torch.int32, device=cu_seqlens.device)


def split_cu_seqlens_for_q_chunk(cu_seqlens_q_rank, T_local, chunk_idx, device):
    T_half = T_local // 2
    chunk_start = chunk_idx * T_half
    chunk_end = (chunk_idx + 1) * T_half

    result = []
    first_doc_spanning = False
    prev_end = 0

    for seq_end in cu_seqlens_q_rank:
        seq_start = prev_end
        prev_end = seq_end
        
        if seq_end <= chunk_start:
            continue
        if seq_start >= chunk_end:
            break

        if seq_start < chunk_start:
            first_doc_spanning = True

        overlap_end = min(seq_end, chunk_end)

        if not result and first_doc_spanning:
            result.append(0)

        result.append(overlap_end - chunk_start)

    if not result:
        result = [T_half]

    if result[0] != 0:
        result.insert(0, 0)

    return torch.tensor(result, dtype=torch.int32, device=device)


def fused_lightning_indexer_kvallgather(
    q,
    k,
    weights,
    index_topk,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout_query='BSND',
    layout_key='BSND',
    layout='BSND',
    cp_group=None,
    cp_stream=None,
    ):
    """
    q: [s, b, n, d] or [T, n, d] (TND)
    k: [s, b, n, d] or [T, 1, d] (TND)
    weights: [s, b, d] or [T, n] (TND)
    index_topk: int
    cp_group: ProcessGroup
    cp_stream: Stream
    """

    cp_size = get_distributed_world_size(cp_group)
    rank = get_distributed_rank(cp_group)
    is_tnd = layout == 'TND'

    if is_tnd:
        # TND: [T, ...] -> [2, T//2, ...] (no transpose, no batch dim)
        q, weights = [
            t.view(2, t.shape[0] // 2, *t.shape[1:])
            for t in [q, weights]
        ]
        # [T, ...] -> [cp*T, ...] (reorder for causal order, no transpose)
        k_ag = gather_and_permute_cp_shard(k, cp_group)

        # Compute cu_seqlens for full gathered K
        cu_seqlens_kv_full = actual_seq_klen * cp_size if cp_size > 1 else actual_seq_klen

        # Compute per-rank Q cu_seqlens
        rank_cu_seqlens_q, _, _ = get_cu_seqlens_qkv_before_attn(actual_seq_qlen, cp_size, rank)
        T_local = q.shape[1]  # T//2 per chunk
    else:
        # BSND: [s, b, ...] -> [2, s//2, b, ...] -> [2, b, s//2, ...]
        q, weights = [
            t.view(2, t.shape[0] // 2, *t.shape[1:]).transpose(1, 2)
            for t in [q, weights]
        ]
        # [s, b, ...] -> [cp*s, b, ...] -> [b, cp*s, ...]
        k_ag = gather_and_permute_cp_shard(k, cp_group).transpose(0, 1)

    indices = [None, None]
    scores = [None, None]
    local_seq_chunk_ids = [rank + 1, 2 * cp_size - rank]
    chunk_size = k_ag.shape[1 if not is_tnd else 0] // cp_size // 2

    for i, chunk_id in enumerate(local_seq_chunk_ids):
        q_ = q[i]
        kv_len = chunk_id * chunk_size

        if is_tnd:
            k_ag_ = k_ag[:kv_len, ...]
            cu_q_chunk = split_cu_seqlens_for_q_chunk(rank_cu_seqlens_q, T_local * 2, i, device=k.device)
            cu_kv_chunk = compute_cu_seqlens_for_window(cu_seqlens_kv_full, kv_len)
        else:
            k_ag_ = k_ag[:, :kv_len, ...]
            cu_q_chunk = actual_seq_qlen
            cu_kv_chunk = actual_seq_klen

        weights_ = weights[i]

        indices[i], scores[i] = torch_npu.npu_lightning_indexer(
            q_,
            k_ag_,
            weights_,
            actual_seq_lengths_query=cu_q_chunk,
            actual_seq_lengths_key=cu_kv_chunk,
            layout_query=layout_query,
            layout_key=layout_key,
            sparse_count=index_topk,
            sparse_mode=3,
            return_value=True,
        )

    if is_tnd:
        # TND: [T//2, N, sparse_size] per step → [T, N, sparse_size]
        topk_indices = torch.cat(indices, dim=0)
        topk_score = torch.cat(scores, dim=0)
    else:
        # BSND: [B, S//2, 1, sparse_size] per step → [B, S, sparse_size]
        topk_indices = torch.cat(indices, dim=1).squeeze(2)
        topk_score = torch.cat(scores, dim=1).squeeze(2)

    return topk_indices, topk_score


def fused_npu_sparse_flash_attention_kvallgather(
    q,
    k,
    v,
    topk_indices,
    q_rope,
    k_rope,
    scale,
    cp_group,
    cp_stream,
    layout='BSND',
    packed_seq_params=None,
    ):
    """
    q: [s, b, n, d] or [T, n, d] (TND)
    k: [s, b, n, d] or [T, n, d] (TND)
    v: [s, b, n, d] or [T, n, d] (TND)
    topk_indices: [b, s, sparse_size] or [T, sparse_size] (TND)
    q_rope: [s, b, n, d] or [T, n, d] (TND)
    k_rope: [s, b, n, d] or [T, n, d] (TND)
    scale: float
    cp_group: ProcessGroup
    cp_stream: Stream
    layout: str ('BSND' or 'TND')
    packed_seq_params: PackedSeqParams (required for TND)
    """

    cp_size = get_distributed_world_size(cp_group)
    rank = get_distributed_rank(cp_group)
    is_tnd = layout == 'TND'

    if scale is None:
        scale = q.shape[-1] ** (-0.5)

    seq_dim = q.shape[0]
    if not (seq_dim % 2 == 0 and k.shape[0] % 2 == 0):
        raise ValueError("Sequence length per GPU needs to be divisible by 2!")

    if is_tnd:
        actual_seq_qlen = packed_seq_params.cu_seqlens_q.to(torch.int32)
        actual_seq_klen = packed_seq_params.cu_seqlens_kv.to(torch.int32)

        # TND: [T, ...] -> [2, T//2, ...] (no transpose)
        q, q_rope = [
            t.view(2, t.shape[0] // 2, *t.shape[1:])
            for t in [q, q_rope]
        ]
        # [T, ...] -> [cp*T, ...] (reorder for causal order, no transpose)
        k_ag = gather_and_permute_cp_shard(k, cp_group)
        v_ag = gather_and_permute_cp_shard(v, cp_group)
        k_rope_ag = gather_and_permute_cp_shard(k_rope, cp_group)

        cu_seqlens_kv_full = actual_seq_klen * cp_size if cp_size > 1 else actual_seq_klen
        rank_cu_seqlens_q, _, _ = get_cu_seqlens_qkv_before_attn(actual_seq_qlen, cp_size, rank)

        # [T, N, sparse_size] -> [2, T//2, N, sparse_size] (no unsqueeze for TND)
        t_total = topk_indices.shape[0]
        topk_indices = topk_indices.view(2, t_total // 2, *topk_indices.shape[1:])
    else:
        # BSND: [s, b, ...] -> [2, s//2, b, ...] -> [2, b, s//2, ...]
        q, q_rope = [
            t.view(2, t.shape[0] // 2, *t.shape[1:]).transpose(1, 2)
            for t in [q, q_rope]
        ]
        # [s, b, ...] -> [cp*s, b, ...] -> [b, cp*s, ...]
        k_ag, v_ag, k_rope_ag = [
            gather_and_permute_cp_shard(t, cp_group).transpose(0, 1)
            for t in [k, v, k_rope]
        ]

        # [b, s, sparse_size] -> [2, b, s//2, 1, sparse_size]
        b, s, sparse_size = topk_indices.shape
        topk_indices = topk_indices.view(b, 2, s // 2, sparse_size).transpose(0, 1).unsqueeze(3)

    out_per_step = [None, None]
    softmax_max = [None, None]
    softmax_sum = [None, None]
    out = torch.empty_like(q)
    local_seq_chunk_ids = [rank + 1, 2 * cp_size - rank]
    chunk_size = k_ag.shape[1 if not is_tnd else 0] // cp_size // 2

    for i, chunk_id in enumerate(local_seq_chunk_ids):
        kv_len = chunk_id * chunk_size

        if is_tnd:
            cu_q_chunk = split_cu_seqlens_for_q_chunk(rank_cu_seqlens_q, seq_dim, i, device=q.device)
            cu_kv_chunk = compute_cu_seqlens_for_window(cu_seqlens_kv_full, kv_len)

            attn_outs = torch_npu.npu_sparse_flash_attention(
                q[i],
                k_ag[:kv_len, ...],
                v_ag[:kv_len, ...],
                sparse_indices=topk_indices[i].to(torch.int32),
                block_table=None,
                actual_seq_lengths_query=cu_q_chunk,
                actual_seq_lengths_kv=cu_kv_chunk,
                query_rope=q_rope[i],
                key_rope=k_rope_ag[:kv_len, ...],
                scale_value=scale,
                sparse_block_size=1,
                layout_query='TND',
                layout_kv='TND',
                sparse_mode=3,
                attention_mode=2,
                return_softmax_lse=True,
            )
        else:
            attn_outs = torch_npu.npu_sparse_flash_attention(
                q[i],
                k_ag[:, :kv_len, ...],
                v_ag[:, :kv_len, ...],
                sparse_indices=topk_indices[i].to(torch.int32),
                block_table=None,
                actual_seq_lengths_query=None,
                actual_seq_lengths_kv=None,
                query_rope=q_rope[i],
                key_rope=k_rope_ag[:, :kv_len, ...],
                scale_value=scale,
                sparse_block_size=1,
                layout_query='BSND',
                layout_kv='BSND',
                sparse_mode=3,
                attention_mode=2,
                return_softmax_lse=True,
            )

        out_per_step[i] = attn_outs[0]
        softmax_max[i] = attn_outs[1]
        softmax_sum[i] = attn_outs[2]

        out[i].copy_(out_per_step[i])

    if is_tnd:
        # TND: [n2, T//2, n1/n2] per step -> [n2, T, n1/n2]
        softmax_max_out = torch.cat(softmax_max, dim=1).contiguous()
        softmax_sum_out = torch.cat(softmax_sum, dim=1).contiguous()
        # [2, T//2, n, d] -> [T, n, d]
        out = out.view(-1, *out.shape[-2:])
    else:
        # BSND: [b, n2, s//2, n1/n2] per step -> [b, n2, s, n1/n2]
        softmax_max_out = torch.cat(softmax_max, dim=2)
        softmax_sum_out = torch.cat(softmax_sum, dim=2)
        # [2, b, s//2, n, d] -> [b, s, n, d] -> [s, b, n, d]
        out = out.transpose(0, 1).contiguous()
        out = out.view(out.shape[0], -1, *out.shape[-2:])
        out = rearrange(out, 'b s h d -> s b h d')

    return out, softmax_max_out, softmax_sum_out


def fused_sparse_lightning_indexer_kl_loss_kvallgather(
    query,
    key,
    query_index,
    key_index,
    weights,
    topk_indices,
    softmax_max,
    softmax_sum,
    scale_value=1,
    *,
    query_rope=None,
    key_rope=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout='BSND',
    sparse_mode=3,
    pre_tokens=65536,
    next_tokens=65536,
    cp_group=None,
    cp_stream=None,
    ):
    """
    query: [s, b, n, d] or [T, n, d] (TND)
    key: [s, b, n, d] or [T, n, d] (TND)
    query_index: [s, b, n, d] or [T, n, d] (TND)
    key_index: [s, b, n, d] or [T, n, d] (TND)
    weights: [s, b, d] or [T, n] (TND)
    topk_indices: [b, s, sparse_size] or [T, sparse_size] (TND)
    softmax_max: [b, n2, s, n1/n2] or [n2, T, n1/n2] (TND)
    softmax_sum: [b, n2, s, n1/n2] or [n2, T, n1/n2] (TND)
    scale_value: float
    query_rope: [s, b, n, d] or [T, n, d] (TND)
    key_rope: [s, b, n, d] or [T, n, d] (TND)
    actual_seq_qlen: Optional[Tensor]
    actual_seq_klen: Optional[Tensor]
    layout: str
    sparse_mode: int
    pre_tokens: int
    next_tokens: int
    cp_group: ProcessGroup
    cp_stream: Stream
    """
    from .dsa_fused import LILossTrain

    cp_size = get_distributed_world_size(cp_group)
    rank = get_distributed_rank(cp_group)
    is_tnd = layout == 'TND'

    sq = query.shape[0]

    if is_tnd:
        actual_seq_qlen = actual_seq_qlen.to(torch.int32) if actual_seq_qlen is not None else None
        actual_seq_klen = actual_seq_klen.to(torch.int32) if actual_seq_klen is not None else None

        # TND: [T, ...] -> [2, T//2, ...] (no transpose)
        query, query_rope, query_index, weights = [
            t.view(2, t.shape[0] // 2, *t.shape[1:])
            for t in [query, query_rope, query_index, weights]
        ]

        # [T, N, sparse_size] -> [2, T//2, N, sparse_size]
        t_total = topk_indices.shape[0]
        topk_indices = topk_indices.view(2, t_total // 2, *topk_indices.shape[1:])

        # [n2, T, n1/n2] -> [n2, 2, T//2, n1/n2] -> [2, n2, T//2, n1/n2]
        softmax_max, softmax_sum = [
            t.view(t.shape[0], 2, t.shape[1] // 2, t.shape[2]).permute(1, 0, 2, 3)
            for t in [softmax_max, softmax_sum]
        ]

        # [T, ...] -> [cp*T, ...] (reorder for causal order, no transpose)
        key_ag = gather_and_permute_cp_shard(key, cp_group)
        key_index_ag = gather_and_permute_cp_shard(key_index, cp_group)
        key_rope_ag = gather_and_permute_cp_shard(key_rope, cp_group)

        cu_seqlens_kv_full = actual_seq_klen * cp_size if cp_size > 1 else actual_seq_klen
        rank_cu_seqlens_q, _, _ = get_cu_seqlens_qkv_before_attn(actual_seq_qlen, cp_size, rank)
        T_local = query.shape[1]  # T//2 per chunk
    else:
        # BSND: [s, b, ...] -> [2, s//2, b, ...] -> [2, b, s//2, ...]
        query, query_rope, query_index, weights = [
            t.view(2, t.shape[0] // 2, *t.shape[1:]).transpose(1, 2)
            for t in [query, query_rope, query_index, weights]
        ]

        # [b, s, sparse_size] -> [b, 2, s//2, sparse_size] -> [2, b, s//2, 1, sparse_size]
        b, s, sparse_size = topk_indices.shape
        topk_indices = topk_indices.view(b, 2, s // 2, sparse_size).transpose(0, 1).unsqueeze(3)

        # [b, 1, s, n] -> [2, b, 1, s//2, n]
        softmax_max, softmax_sum = [
            rearrange(t, 'b n2 (c s) n1 -> c b n2 s n1', n2=1, c=2)
            for t in [softmax_max, softmax_sum]
        ]

        # [s, b, ...] -> [cp*s, b, ...] -> [b, cp*s, ...]
        key_ag, key_index_ag, key_rope_ag = [
            gather_and_permute_cp_shard(t, cp_group).transpose(0, 1)
            for t in [key, key_index, key_rope]
        ]

    loss = [None, None]
    local_seq_chunk_ids = [rank + 1, 2 * cp_size - rank]
    chunk_size = key_ag.shape[1 if not is_tnd else 0] // cp_size // 2

    for i, chunk_id in enumerate(local_seq_chunk_ids):
        kv_len = chunk_id * chunk_size

        if is_tnd:
            cu_q_chunk = split_cu_seqlens_for_q_chunk(rank_cu_seqlens_q, T_local * 2, i, device=query.device)
            cu_kv_chunk = compute_cu_seqlens_for_window(cu_seqlens_kv_full, kv_len)

            loss[i] = LILossTrain.apply(
                query[i],
                key_ag[:kv_len, ...],
                query_index[i],
                key_index_ag[:kv_len, ...],
                weights[i],
                topk_indices[i],
                softmax_max[i],
                softmax_sum[i],
                scale_value,
                query_rope[i],
                key_rope_ag[:kv_len, ...],
                cu_q_chunk,
                cu_kv_chunk,
                'TND',
                sparse_mode,
                pre_tokens,
                next_tokens,
            )
        else:
            loss[i] = LILossTrain.apply(
                query[i],
                key_ag[:, :kv_len, ...],
                query_index[i],
                key_index_ag[:, :kv_len, ...],
                weights[i],
                topk_indices[i],
                softmax_max[i],
                softmax_sum[i],
                scale_value,
                query_rope[i],
                key_rope_ag[:, :kv_len, ...],
                actual_seq_qlen,
                actual_seq_klen,
                'BSND',
                sparse_mode,
                pre_tokens,
                next_tokens,
            )

    return sum(loss) / sq


def transformer_config_post_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self):
        #Reset context_parallel_size to bypass Megatron dev check.
        ori_context_parallel_size = self.context_parallel_size
        self.context_parallel_size = 1
        fn(self)
        self.context_parallel_size = ori_context_parallel_size
        del ori_context_parallel_size

    return wrapper
