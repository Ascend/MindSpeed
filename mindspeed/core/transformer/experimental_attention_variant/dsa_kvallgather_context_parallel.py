# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from functools import wraps

import torch
import torch_npu
from einops import rearrange

from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from mindspeed.te.pytorch.attention.dot_product_attention.kvallgather_context_parallel import (
    get_seq_chunk_ids_for_reordering_before_attn
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


def fused_lightning_indexer_kvallgather(
    q,
    k,
    weights,
    index_topk,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout_query='BSND',
    layout_key='BSND',
    cp_group=None,
    cp_stream=None,
    ):
    """
    q: [s, b, n, d]
    k: [s, b, n, d]
    weights: [s, b, d]
    index_topk: int
    cp_group: ProcessGroup
    cp_stream: Stream
    """

    cp_size = get_distributed_world_size(cp_group)
    rank = get_distributed_rank(cp_group)

    # [s, b, ...] -> [2, s//2, b, ...] -> [2, b, s//2, ...]
    q, weights = [
        t.view(2, t.shape[0] // 2, *t.shape[1:]).transpose(1, 2)
        for t in [q, weights]
    ]

    # [s, b, ...] -> [cp*s, b, ...] -> [b, cp*s, ...]
    k_ag = gather_and_permute_cp_shard(k, cp_group).transpose(0, 1)

    indices = [None, None]
    scores = [None, None]
    local_seq_chunk_ids = [rank + 1, 2 * cp_size - rank]
    chunk_size = k_ag.shape[1] // cp_size // 2

    stream_list = [torch.npu.current_stream(), cp_stream]
    cp_stream.wait_stream(torch.npu.current_stream())

    for i, chunk_id in enumerate(local_seq_chunk_ids):
        with torch.npu.stream(stream_list[i]):
            q_ = q[i].contiguous()
            k_ag_ = k_ag[:, :chunk_id * chunk_size, ...].contiguous()

            weights_ = weights[i].contiguous()

            indices[i], scores[i] = torch_npu.npu_lightning_indexer(
                q_,
                k_ag_,
                weights_,
                actual_seq_lengths_query=actual_seq_qlen,
                actual_seq_lengths_key=actual_seq_klen,
                layout_query=layout_query,
                layout_key=layout_key,
                sparse_count=index_topk,
                sparse_mode=3,
                return_value=True,
            )

    torch.npu.current_stream().wait_stream(cp_stream)

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
    cp_stream
    ):
    """
    q: [s, b, n, d]
    k: [s, b, n, d]
    v: [s, b, n, d]
    topk_indices: [b, s, sparse_size]
    q_rope: [s, b, n, d]
    k_rope: [s, b, n, d]
    scale: float
    cp_group: ProcessGroup
    cp_stream: Stream
    """

    if scale is None:
        scale = q.shape[-1] ** (-0.5)

    if not (q.shape[0] % 2 == 0 and k.shape[0] % 2 == 0):
        raise AssertionError("Sequence length per GPU needs to be divisible by 2!")

    # [s, b, ...] -> [2, s//2, b, ...] -> [2, b, s//2, ...]
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

    # create two streams to resolve wave quantization issue of Flash Attn in each step
    cp_stream.wait_stream(torch.npu.current_stream())
    flash_attn_streams = [torch.npu.current_stream(), cp_stream]

    out_per_step = [None, None]
    softmax_max = [None, None]
    softmax_sum = [None, None]
    # [2, b, s//2, n, d]
    out = torch.empty_like(q)

    num_steps = 2
    for i in range(num_steps + 1):
        if i < num_steps:
            with torch.npu.stream(flash_attn_streams[i]):
                # [b, 2, s//2, n, d] -> [b, s//2, n, d]
                q_ = q[i].contiguous()
                q_rope_ = q_rope[i].contiguous()
                topk_indices_ = topk_indices[i].contiguous()

                attn_outs = torch_npu.npu_sparse_flash_attention(
                    q_, k_ag, v_ag,
                    sparse_indices=topk_indices_.to(torch.int32),
                    block_table=None,
                    actual_seq_lengths_query=None,
                    actual_seq_lengths_kv=None,
                    query_rope=q_rope_,
                    key_rope=k_rope_ag,
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

        if i > 0:
            with torch.npu.stream(flash_attn_streams[i - 1]):
                out[i - 1].copy_(out_per_step[i - 1])

    torch.npu.current_stream().wait_stream(cp_stream)

    # [b, n2, s, n1/n2]
    softmax_max_out = torch.cat(softmax_max, dim=2)
    softmax_sum_out = torch.cat(softmax_sum, dim=2)
    # [2, b, s//2, n, d] -> [b, s, n, d]
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
    query: [s, b, n, d]
    key: [s, b, n, d]
    query_index: [s, b, n, d]
    key_index: [s, b, n, d]
    weights: [s, b, d]
    topk_indices: [b, s, sparse_size]
    softmax_max: [b, n2, s, n1/n2]
    softmax_sum: [b, n2, s, n1/n2]
    scale_value: float
    query_rope: [s, b, n, d]
    key_rope: [s, b, n, d]
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

    sq = query.shape[0]

    # [s, b, ...] -> [2, s//2, b, ...] -> [2, b, s//2, ...]
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
    chunk_size = key_ag.shape[1] // cp_size // 2

    stream_list = [torch.npu.current_stream(), cp_stream]
    cp_stream.wait_stream(torch.npu.current_stream())

    for i, chunk_id in enumerate(local_seq_chunk_ids):
        with torch.npu.stream(stream_list[i]):
            kv_len = chunk_id * chunk_size
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
                layout,
                sparse_mode,
                pre_tokens,
                next_tokens,
            )

    torch.npu.current_stream().wait_stream(cp_stream)

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