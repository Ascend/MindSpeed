# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2022-2025; NVIDIA CORPORATION AFFILIATES.
# Copyright c) 2022-2025 Advanced Micro Devices Inc.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch_npu

from .utils import get_distributed_rank, get_distributed_world_size

_seq_chunk_ids_cache_for_reordering_before_attn = {}
_seq_chunk_ids_cache_for_reordering_after_attn = {}
_THD_LOAD_BALANCED_CP_METADATA_CACHE_SIZE = 8
_thd_load_balanced_cp_metadata_cache = {}


def gather_along_first_dim(
    inp: torch.Tensor,
    process_group,
    async_op: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    """
    All-gather tensors and concatenate along first dimension.
    """
    world_size = get_distributed_world_size(process_group)
    if world_size == 1:
        return inp, None

    out_shape = list(inp.size())
    out_shape[0] *= world_size

    # Communication for plain PyTorch tensors
    out = torch.empty(
        out_shape,
        dtype=inp.dtype,
        device=inp.device,
        memory_format=torch.contiguous_format,
    )
    handle = torch.distributed.all_gather_into_tensor(
        out,
        inp.contiguous(),
        group=process_group,
        async_op=async_op,
    )
    return out, handle


def reduce_scatter_along_first_dim(
    inp: torch.Tensor, process_group, async_op: bool = False
) -> Tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_distributed_world_size(process_group)
    # Bypass the function if we are using only 1 NPU.
    if world_size == 1:
        return inp, None

    dim_size = list(inp.size())
    assert dim_size[0] % world_size == 0, "First dimension must be divisible by world size"

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(dim_size, dtype=inp.dtype, device=inp.device)
    handle = torch.distributed.reduce_scatter_tensor(output, inp.contiguous(), group=process_group, async_op=async_op)
    return output, handle


def get_seq_chunk_ids_for_reordering_before_attn(cp_size, device):
    """
    Context parallelism assigns two discontiguous sequence chunks to each NPU for load balancing.
    To make sure tokens are ordered correctly for compute, we need to reorder sequence chunks to
    be contigupus before attention compute. This function is to compute sequence chunk ids for
    reordering.
    """
    if (cp_size, device) not in _seq_chunk_ids_cache_for_reordering_before_attn:
        chunk_ids = torch.empty(2 * cp_size, dtype=torch.int32, device=device)
        for rank in range(cp_size):
            chunk_ids[rank] = 2 * rank
            chunk_ids[rank + cp_size] = 2 * cp_size - 2 * rank - 1
        _seq_chunk_ids_cache_for_reordering_before_attn[(cp_size, device)] = chunk_ids
    return _seq_chunk_ids_cache_for_reordering_before_attn[(cp_size, device)]


def get_seq_chunk_ids_for_reordering_after_attn(cp_size, device):
    """
    Context parallelism assigns two discontiguous sequence chunks to each NPU for load balancing.
    We need to reorder sequence chunks back to discontiguous after attention compute. This function
    is to compute sequence chunk ids for reordering.
    """
    if (cp_size, device) not in _seq_chunk_ids_cache_for_reordering_after_attn:
        chunk_ids = torch.empty(2 * cp_size, dtype=torch.int32, device=device)
        for rank in range(cp_size):
            chunk_ids[2 * rank] = rank
            chunk_ids[2 * rank + 1] = 2 * cp_size - rank - 1
        _seq_chunk_ids_cache_for_reordering_after_attn[(cp_size, device)] = chunk_ids
    return _seq_chunk_ids_cache_for_reordering_after_attn[(cp_size, device)]


def get_kv_seq_info_after_all_gather(local_chunk_id, cp_size, max_seqlen_kv, causal):
    """Compute KV sequence index range and update window size after all-gather."""
    local_chunk_end_idx = (local_chunk_id + 1) * max_seqlen_kv
    full_seq_end_idx = max_seqlen_kv * cp_size * 2

    seq_start_idx = 0

    if causal:
        seq_end_idx = local_chunk_end_idx
    else:
        seq_end_idx = full_seq_end_idx

    return (seq_start_idx, seq_end_idx)


class AttnFuncWithCPAndKVAllGatherForSBHD(torch.autograd.Function):
    """
    Attention implementation with context parallelism. KV all-gather between CP ranks is exposed.
    For SBHD format (SBH shape_order)
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        n_head,
        attention_mask,
        qkv_format,
        attn_mask_type,
        attention_dropout,
        softmax_scale,
        deterministic,
        cp_group,
        cp_stream,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)

        causal = 'causal' in attn_mask_type
        if not causal:
            raise AssertionError("Only causal mask is supported for AllGatherContextParallel.")

        seq_dim = qkv_format.index("s")
        if not (q.shape[seq_dim] % 2 == 0 and k.shape[seq_dim] % 2 == 0):
            raise AssertionError("Sequence length per GPU needs to be divisible by 2!")

        # [s, b, h] -> [2, s//2, b, h]
        q = q.view(2, q.shape[seq_dim] // 2, *q.shape[(seq_dim + 1) :])

        # [s, b, h] -> [cp, s, b, h]
        k_ag, _ = gather_along_first_dim(k, cp_group)
        v_ag, _ = gather_along_first_dim(v, cp_group)

        # [cp, s, b, h] -> [cp*2, s//2, b, h]
        k_ag = k_ag.view(2 * cp_size, k.shape[0] // 2, *k.shape[1:])
        v_ag = v_ag.view(2 * cp_size, v.shape[0] // 2, *v.shape[1:])
        chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering_before_attn(cp_size, k.device)
        k_ag = torch.index_select(k_ag, dim=0, index=chunk_ids_for_kv_ag)
        v_ag = torch.index_select(v_ag, dim=0, index=chunk_ids_for_kv_ag)
        # [cp*2, s//2, b, h] -> [cp*s, b, h]
        k_ag = k_ag.view(-1, *k.shape[1:])
        v_ag = v_ag.view(-1, *v.shape[1:])
        kv_len = k_ag.shape[0] // (2 * cp_size)
        cp_stream.wait_stream(torch.npu.current_stream())

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.npu.current_stream(), cp_stream]

        local_seq_chunk_ids = [rank, 2 * cp_size - rank - 1]
        kv_seq_range_per_step = [None, None]
        out_per_step = [None, None]
        softmax_max = [None, None]
        softmax_sum = [None, None]
        # [2, s//2, b, h]
        if k.shape[-1] == v.shape[-1]:
            out = torch.empty_like(q)
        else:
            out = torch.empty(*q.shape[:-1], v.shape[-1], dtype=q.dtype, device=q.device)

        for i in range(len(local_seq_chunk_ids) + 1):
            if i < len(local_seq_chunk_ids):
                with torch.npu.stream(flash_attn_streams[i]):
                    # [2, sq//2, b, h] -> [sq//2, b, h]
                    q_ = q.select(seq_dim, i).contiguous()
                    kv_seq_range_per_step[i] = get_kv_seq_info_after_all_gather(
                        local_seq_chunk_ids[i],
                        cp_size,
                        kv_len,
                        causal,
                    )
                    seq_start_idx, seq_end_idx = (
                        kv_seq_range_per_step[i][0],
                        kv_seq_range_per_step[i][1],
                    )
                    # [cp*s, b, h] -> [s_range, b, h]
                    k_, v_ = [x[seq_start_idx:seq_end_idx] for x in [k_ag, v_ag]]

                    attn_outs = torch_npu.npu_fusion_attention(
                        q_,
                        k_,
                        v_,
                        n_head,
                        'SBH',
                        pse=None,
                        padding_mask=None,
                        atten_mask=attention_mask,
                        scale=softmax_scale,
                        pre_tockens=65536,
                        next_tockens=0,
                        keep_prob=1 - attention_dropout,
                        inner_precise=0,
                        sparse_mode=3,
                    )
                    out_per_step[i] = attn_outs[0]
                    softmax_max[i] = attn_outs[1]
                    softmax_sum[i] = attn_outs[2]

            if i > 0:
                with torch.npu.stream(flash_attn_streams[i - 1]):
                    out[i - 1].copy_(out_per_step[i - 1])

        torch.npu.current_stream().wait_stream(cp_stream)

        # [2, s//2, b, h] -> [s, b, h]
        out = out.view(-1, *out.shape[-2:])
        # [2, s//2, b, h] -> [s, b, h]
        q = q.view(-1, *q.shape[2:])

        ctx.save_for_backward(q, k, v, *out_per_step, *softmax_max, *softmax_sum)

        ctx.kv_seq_range_per_step = kv_seq_range_per_step
        ctx.cp_group = cp_group
        ctx.cp_stream = cp_stream
        ctx.qkv_format = qkv_format
        ctx.n_head = n_head
        ctx.attention_dropout = attention_dropout
        ctx.softmax_scale = softmax_scale
        ctx.attention_mask = attention_mask
        return out

    @staticmethod
    def backward(ctx, dout):
        cp_size = get_distributed_world_size(ctx.cp_group)
        rank = get_distributed_rank(ctx.cp_group)

        (*saved_tensors,) = ctx.saved_tensors
        (q, k, v) = saved_tensors[:3]
        out_per_step = saved_tensors[3:5]
        softmax_max = saved_tensors[5:7]
        softmax_sum = saved_tensors[7:9]
        kv_seq_range_per_step = ctx.kv_seq_range_per_step

        seq_dim = ctx.qkv_format.index("s")

        # [s, b, h] -> [2, s//2, b, h]
        q = q.view(2, q.shape[seq_dim] // 2, *q.shape[(seq_dim + 1) :])
        # [s, b, h_v] -> [2, s//2, b, h_v]
        if k.shape[-1] == v.shape[-1]:
            dout = dout.view(q.shape)
        else:
            dout = dout.view(*q.shape[:-1], v.shape[-1])

        # [2, s//2, b, h]
        dq = torch.empty_like(q)
        # [cp*s, b, h]
        dk = torch.zeros((k.shape[0] * cp_size, *k.shape[1:]), dtype=k.dtype, device=k.device)
        dv = torch.zeros((v.shape[0] * cp_size, *v.shape[1:]), dtype=v.dtype, device=v.device)
        dq_per_step = [None, None]
        dk_per_step = [None, None]
        dv_per_step = [None, None]

        # create two streams for Flash Attn
        flash_attn_streams = [torch.npu.current_stream(), ctx.cp_stream]
        # synchronize dkv update across steps
        dkv_update_done = torch.npu.Event()

        # [s, b, h] -> [cp, s, b, h]
        k_ag, _ = gather_along_first_dim(k, ctx.cp_group)
        v_ag, _ = gather_along_first_dim(v, ctx.cp_group)

        # [cp, s, b, h] -> [cp*2, s//2, b, h]
        k_ag = k_ag.view(2 * cp_size, k.shape[0] // 2, *k.shape[1:])
        v_ag = v_ag.view(2 * cp_size, v.shape[0] // 2, *v.shape[1:])
        chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering_before_attn(cp_size, k.device)
        k_ag = torch.index_select(k_ag, dim=0, index=chunk_ids_for_kv_ag)
        v_ag = torch.index_select(v_ag, dim=0, index=chunk_ids_for_kv_ag)
        # [cp*2, s//2, b, h] -> [cp*s, b, h]
        k_ag = k_ag.view(-1, *k.shape[1:])
        v_ag = v_ag.view(-1, *v.shape[1:])
        ctx.cp_stream.wait_stream(torch.npu.current_stream())

        local_seq_chunk_ids = [rank, 2 * cp_size - rank - 1]

        for i in range(len(local_seq_chunk_ids) + 1):
            if i < len(local_seq_chunk_ids):
                with torch.npu.stream(flash_attn_streams[i]):
                    # [2, sq//2, b, h] -> [sq//2, b, h]
                    q_ = q.select(seq_dim, i).contiguous()
                    seq_start_idx, seq_end_idx = (
                        kv_seq_range_per_step[i][0],
                        kv_seq_range_per_step[i][1],
                    )
                    k_, v_ = [x[seq_start_idx:seq_end_idx] for x in [k_ag, v_ag]]

                    out_ = out_per_step[i]
                    # [2, s//2, b, h]  -> [s//2, b, h]
                    dout_ = dout.select(seq_dim, i).contiguous()

                    attn_grad_outs = torch_npu.npu_fusion_attention_grad(
                        q_,
                        k_,
                        v_,
                        dout_,
                        ctx.n_head,
                        "SBH",
                        pse=None,
                        padding_mask=None,
                        atten_mask=ctx.attention_mask,
                        softmax_max=softmax_max[i],
                        softmax_sum=softmax_sum[i],
                        attention_in=out_,
                        scale_value=ctx.softmax_scale,
                        pre_tockens=65536,
                        next_tockens=0,
                        sparse_mode=3,
                        keep_prob=1 - ctx.attention_dropout,
                    )
                    dq_per_step[i] = attn_grad_outs[0]
                    dk_per_step[i] = attn_grad_outs[1]
                    dv_per_step[i] = attn_grad_outs[2]

            if i > 0:
                with torch.npu.stream(flash_attn_streams[i - 1]):
                    dq[i - 1].copy_(dq_per_step[i - 1])

                    # wait until dkv update of last step is done
                    if i > 1:
                        flash_attn_streams[i - 1].wait_event(dkv_update_done)
                    seq_start_idx, seq_end_idx = (
                        kv_seq_range_per_step[i - 1][0],
                        kv_seq_range_per_step[i - 1][1],
                    )
                    dk[seq_start_idx:seq_end_idx].add_(dk_per_step[i - 1])
                    dv[seq_start_idx:seq_end_idx].add_(dv_per_step[i - 1])
                    if i < len(local_seq_chunk_ids):
                        flash_attn_streams[i - 1].record_event(dkv_update_done)

        torch.npu.current_stream().wait_stream(ctx.cp_stream)

        # [cp*s, b, h] -> [cp*2, s//2, b, h]
        dk = dk.view(2 * cp_size, -1, *dk.shape[-2:])
        dv = dv.view(2 * cp_size, -1, *dv.shape[-2:])
        chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering_after_attn(cp_size, dk.device)
        dk = torch.index_select(dk, dim=0, index=chunk_ids_for_kv_ag)
        dv = torch.index_select(dv, dim=0, index=chunk_ids_for_kv_ag)
        # [cp*2, s//2, b, h] -> [cp*s, b, h]
        dk = dk.view(-1, *dk.shape[-2:])
        dv = dv.view(-1, *dv.shape[-2:])
        dk, _ = reduce_scatter_along_first_dim(dk, ctx.cp_group)
        dv, _ = reduce_scatter_along_first_dim(dv, ctx.cp_group)

        # [2, s//2, b, h] -> [s, b, h]
        dq = dq.view(-1, *dq.shape[-2:])

        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def get_thd_load_balanced_cp_metadata(cu_seqlens, cp_size, rank, device):
    """Build token indices for EOD load-balanced KVAllGather THD attention.

    Each packed subsequence is divided into ``2 * cp_size`` chunks. Rank ``r`` owns
    chunk ``r`` and chunk ``2 * cp_size - r - 1``. K/V all-gather produces a
    rank-major tensor, while TND attention requires each logical sequence to be
    contiguous, so this helper builds direct rank-major indices for the two
    sequence-major causal KV prefixes used by the local Q chunks.
    """
    if cp_size < 1:
        raise AssertionError("context_parallel_size must be greater than zero.")
    if rank < 0 or rank >= cp_size:
        raise AssertionError("context parallel rank must be in [0, context_parallel_size).")

    cu_seqlens = tuple(int(seq_len) for seq_len in cu_seqlens)
    if not cu_seqlens:
        raise AssertionError("cu_seqlens must contain at least one sequence endpoint.")

    cache_key = (cp_size, rank, device)
    cached_metadata = _thd_load_balanced_cp_metadata_cache.get(cache_key)
    if cached_metadata is not None and cached_metadata["cu_seqlens"] == cu_seqlens:
        return cached_metadata["metadata"]

    seq_starts = (0,) + cu_seqlens[:-1]
    seq_lens = [end - start for start, end in zip(seq_starts, cu_seqlens)]
    if any(seq_len <= 0 for seq_len in seq_lens):
        raise AssertionError("Each packed subsequence must have a positive sequence length.")
    if any(seq_len % (2 * cp_size) != 0 for seq_len in seq_lens):
        raise AssertionError("Each packed subsequence length must be divisible by 2 * context_parallel_size.")

    chunk_lens = [seq_len // (2 * cp_size) for seq_len in seq_lens]
    local_seq_starts = []
    local_total_len = 0
    for chunk_len in chunk_lens:
        local_seq_starts.append(local_total_len)
        local_total_len += 2 * chunk_len

    full_total_len = cu_seqlens[-1]
    if local_total_len * cp_size != full_total_len:
        raise AssertionError("Packed sequence length is inconsistent with context parallel size.")

    kv_index_in_rank_major = []
    actual_seq_qlen = []
    actual_seq_kvlen = []
    local_chunk_ids = (rank, 2 * cp_size - rank - 1)
    q_end = 0
    kv_end = 0

    for local_seq_start, chunk_len in zip(local_seq_starts, chunk_lens):
        for chunk_id in local_chunk_ids:
            kv_prefix_len = (chunk_id + 1) * chunk_len
            for prefix_chunk_id in range(chunk_id + 1):
                if prefix_chunk_id < cp_size:
                    source_rank = prefix_chunk_id
                    source_slot = 0
                else:
                    source_rank = 2 * cp_size - prefix_chunk_id - 1
                    source_slot = 1
                source_start = source_rank * local_total_len + local_seq_start + source_slot * chunk_len
                kv_index_in_rank_major.extend(range(source_start, source_start + chunk_len))

            q_end += chunk_len
            kv_end += kv_prefix_len
            actual_seq_qlen.append(q_end)
            actual_seq_kvlen.append(kv_end)

    kv_index_in_rank_major = torch.tensor(kv_index_in_rank_major, dtype=torch.long, device=device)

    metadata = {
        "cu_seqlens": cu_seqlens,
        "local_total_len": local_total_len,
        "full_total_len": full_total_len,
        "kv_index_in_rank_major": kv_index_in_rank_major,
        "actual_seq_qlen": actual_seq_qlen,
        "actual_seq_kvlen": actual_seq_kvlen,
    }
    _thd_load_balanced_cp_metadata_cache[cache_key] = {
        "cu_seqlens": cu_seqlens,
        "metadata": metadata,
    }
    return metadata


@lru_cache(maxsize=_THD_LOAD_BALANCED_CP_METADATA_CACHE_SIZE)
def _get_thd_load_balanced_cp_metadata_for_tensor(cu_seqlens, _tensor_version, cp_size, rank, device):
    """Reuse metadata across layers that share the same cu_seqlens tensor."""
    return get_thd_load_balanced_cp_metadata(cu_seqlens.tolist(), cp_size, rank, device)


def clear_thd_load_balanced_cp_metadata_cache():
    """Clear tensor and metadata caches, primarily for tests."""
    _get_thd_load_balanced_cp_metadata_for_tensor.cache_clear()
    _thd_load_balanced_cp_metadata_cache.clear()


def _get_thd_load_balanced_cp_metadata_cached(cu_seqlens, cp_size, rank, device):
    """Avoid repeated cu_seqlens conversion for Transformer layers in one microbatch."""
    if isinstance(cu_seqlens, torch.Tensor):
        try:
            tensor_version = cu_seqlens._version
        except RuntimeError:
            tensor_version = None
        if tensor_version is not None:
            return _get_thd_load_balanced_cp_metadata_for_tensor(cu_seqlens, tensor_version, cp_size, rank, device)
        cu_seqlens = cu_seqlens.tolist()
    return get_thd_load_balanced_cp_metadata(cu_seqlens, cp_size, rank, device)


def _validate_thd_attention_shapes(q, k, v, n_head):
    """Validate TND shapes for MHA, GQA and expanded MLA attention."""
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise AssertionError("TND Q, K and V must all be three-dimensional tensors.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise AssertionError("Q, K and V must have the same dtype.")
    if q.shape[1] != n_head:
        raise AssertionError("n_head must match the number of query heads for TND attention.")
    if k.shape[1] != v.shape[1]:
        raise AssertionError("K and V must have the same number of heads.")
    if q.shape[1] == 0 or k.shape[1] == 0 or q.shape[1] % k.shape[1] != 0:
        raise AssertionError("The number of query heads must be divisible by the number of KV heads.")
    if q.shape[-1] != k.shape[-1]:
        raise AssertionError("Q and K must have the same head dimension.")
    if k.shape[-1] < v.shape[-1]:
        raise AssertionError("The K head dimension must be greater than or equal to the V head dimension.")


class AttnFuncWithCPAndKVAllGatherForTHD(torch.autograd.Function):
    """KVAllGather THD attention with EOD-aware causal load balancing."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        n_head,
        attention_mask,
        qkv_format,
        attn_mask_type,
        attention_dropout,
        softmax_scale,
        deterministic,
        cp_group,
        cu_seqlens_q,
        cu_seqlens_kv,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)

        if 'causal' not in attn_mask_type:
            raise AssertionError("Only causal mask is supported for AllGatherContextParallel.")
        _validate_thd_attention_shapes(q, k, v, n_head)
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise AssertionError("Q, K and V must have the same local token count for THD self-attention.")

        metadata = _get_thd_load_balanced_cp_metadata_cached(cu_seqlens_q, cp_size, rank, q.device)
        if cu_seqlens_q is not cu_seqlens_kv:
            kv_metadata = _get_thd_load_balanced_cp_metadata_cached(cu_seqlens_kv, cp_size, rank, q.device)
            if kv_metadata["cu_seqlens"] != metadata["cu_seqlens"]:
                raise AssertionError("cu_seqlens_q and cu_seqlens_kv must be the same for THD format.")
        if q.shape[0] != metadata["local_total_len"]:
            raise AssertionError("The local THD token count does not match the EOD load-balanced CP partition.")

        k_ag, _ = gather_along_first_dim(k, cp_group)
        v_ag, _ = gather_along_first_dim(v, cp_group)
        kv_index = metadata["kv_index_in_rank_major"]
        k_attn = k_ag.index_select(0, kv_index)
        v_attn = v_ag.index_select(0, kv_index)

        attn_outs = torch_npu.npu_fusion_attention(
            q,
            k_attn,
            v_attn,
            n_head,
            'TND',
            pse=None,
            padding_mask=None,
            atten_mask=attention_mask,
            scale=softmax_scale,
            pre_tockens=65536,
            next_tockens=0,
            keep_prob=1 - attention_dropout,
            inner_precise=0,
            sparse_mode=3,
            actual_seq_qlen=metadata["actual_seq_qlen"],
            actual_seq_kvlen=metadata["actual_seq_kvlen"],
        )
        out, softmax_max, softmax_sum = attn_outs[:3]

        ctx.save_for_backward(q, k, v, out, softmax_max, softmax_sum)
        ctx.cp_group = cp_group
        ctx.n_head = n_head
        ctx.attention_dropout = attention_dropout
        ctx.softmax_scale = softmax_scale
        ctx.attention_mask = attention_mask
        ctx.kv_index_in_rank_major = metadata["kv_index_in_rank_major"]
        ctx.actual_seq_qlen = metadata["actual_seq_qlen"]
        ctx.actual_seq_kvlen = metadata["actual_seq_kvlen"]
        return out

    @staticmethod
    def backward(ctx, dout):
        cp_size = get_distributed_world_size(ctx.cp_group)
        saved_tensors = ctx.saved_tensors
        q, k, v, out, softmax_max, softmax_sum = saved_tensors

        k_ag, _ = gather_along_first_dim(k, ctx.cp_group)
        v_ag, _ = gather_along_first_dim(v, ctx.cp_group)
        kv_index = ctx.kv_index_in_rank_major
        k_attn = k_ag.index_select(0, kv_index)
        v_attn = v_ag.index_select(0, kv_index)

        dk = torch.zeros((k.shape[0] * cp_size, *k.shape[1:]), dtype=k.dtype, device=k.device)
        dv = torch.zeros((v.shape[0] * cp_size, *v.shape[1:]), dtype=v.dtype, device=v.device)

        attn_grad_outs = torch_npu.npu_fusion_attention_grad(
            q,
            k_attn,
            v_attn,
            dout,
            ctx.n_head,
            "TND",
            pse=None,
            padding_mask=None,
            atten_mask=ctx.attention_mask,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            attention_in=out,
            scale_value=ctx.softmax_scale,
            pre_tockens=65536,
            next_tockens=0,
            sparse_mode=3,
            keep_prob=1 - ctx.attention_dropout,
            actual_seq_qlen=ctx.actual_seq_qlen,
            actual_seq_kvlen=ctx.actual_seq_kvlen,
        )
        dq = attn_grad_outs[0]
        dk.index_add_(0, kv_index, attn_grad_outs[1])
        dv.index_add_(0, kv_index, attn_grad_outs[2])

        dk, _ = reduce_scatter_along_first_dim(dk, ctx.cp_group)
        dv, _ = reduce_scatter_along_first_dim(dv, ctx.cp_group)

        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
