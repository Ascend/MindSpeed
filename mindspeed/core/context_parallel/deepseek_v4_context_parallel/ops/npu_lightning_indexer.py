# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""Lazy wrapper for the CANN Lightning Indexer operator."""

from functools import lru_cache

import torch

from .._utils import normalize_cu_seqlens


_CUSTOM_OPS = None


def _custom_ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is not None:
        return _CUSTOM_OPS
    try:
        import cann_ops_transformer.ops as custom_ops
    except ImportError:
        custom_ops = None
    _CUSTOM_OPS = custom_ops
    return _CUSTOM_OPS


@lru_cache(maxsize=16)
def _metadata_cached(
    batch_size,
    seq_q,
    heads_q,
    head_dim,
    seq_k,
    heads_k,
    layout,
    cu_q_values,
    cu_k_values,
    topk,
    sparse_mode,
    cmp_ratio,
    residual_values,
    device,
):
    custom_ops = _custom_ops()
    if custom_ops is None:
        raise RuntimeError("Lightning Indexer requires the cann_ops_transformer extension.")
    cu_q = None if cu_q_values is None else torch.tensor(cu_q_values, dtype=torch.int32, device=device)
    cu_k = None if cu_k_values is None else torch.tensor(cu_k_values, dtype=torch.int32, device=device)
    residual = torch.tensor(residual_values, dtype=torch.int32, device=device)
    return custom_ops.lightning_indexer_metadata(
        heads_q,
        heads_k,
        head_dim,
        topk,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        cmp_residual_k=residual,
        batch_size=batch_size,
        max_seqlen_q=seq_q,
        max_seqlen_k=seq_k,
        layout_q=layout,
        layout_k=layout,
        mask_mode=sparse_mode,
        cmp_ratio=cmp_ratio,
    )


def get_npu_lightning_indexer_metadata(
    B,
    S_Q,
    N1,
    D,
    S_K,
    N2,
    layout="BSND",
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    topk=128,
    sparse_mode=3,
    cmp_ratio=4,
    cmp_residual_k_values=None,
    device=None,
):
    if device is None:
        device = torch.device("npu")
    if cmp_residual_k_values is None:
        cmp_residual_k_values = (int(S_K) % int(cmp_ratio),) * int(B)
    residual_values = tuple(int(value) for value in cmp_residual_k_values)
    metadata = _metadata_cached(
        int(B),
        int(S_Q),
        int(N1),
        int(D),
        int(S_K),
        int(N2),
        layout,
        _cu_values_for_metadata_cache(cu_seqlens_q),
        _cu_values_for_metadata_cache(cu_seqlens_k),
        int(topk),
        int(sparse_mode),
        int(cmp_ratio),
        residual_values,
        str(device),
    )
    return metadata, torch.tensor(residual_values, dtype=torch.int32, device=device)


def _cu_values_for_metadata_cache(cu_seqlens):
    if cu_seqlens is None:
        return None
    if torch.is_tensor(cu_seqlens):
        cu = normalize_cu_seqlens(cu_seqlens, cu_seqlens.device, name="Lightning Indexer cu_seqlens")
        return tuple(int(value) for value in cu.detach().cpu().tolist())
    cu = normalize_cu_seqlens(cu_seqlens, torch.device("cpu"), name="Lightning Indexer cu_seqlens")
    return tuple(int(value) for value in cu.tolist())


def npu_lightning_indexer(
    query,
    key,
    weights,
    topk,
    layout="BSND",
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    cmp_residual_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    sparse_mode=3,
    cmp_ratio=4,
    return_values=True,
):
    custom_ops = _custom_ops()
    if custom_ops is None:
        raise RuntimeError("Lightning Indexer requires the cann_ops_transformer extension.")
    if layout not in ("BSND", "TND"):
        raise ValueError(f"Unsupported Lightning Indexer layout: {layout}.")
    if query.dim() != 4 or key.dim() != 4 or weights.dim() != 3:
        raise ValueError("Lightning Indexer inputs must use model layout [S, B, N, D] and [S, B, N].")

    seq_q, batch_size, heads_q, head_dim = query.shape
    seq_k, key_batch, heads_k, key_dim = key.shape
    if key_batch != batch_size or key_dim != head_dim:
        raise ValueError("Lightning Indexer query/key batch and head dimensions must match.")
    if tuple(weights.shape) != (seq_q, batch_size, heads_q):
        raise ValueError("Lightning Indexer weights must have shape [S_Q, B, N_Q].")

    cu_q = normalize_cu_seqlens(cu_seqlens_q, query.device, name="Lightning Indexer cu_seqlens")
    cu_k = normalize_cu_seqlens(cu_seqlens_k, query.device, name="Lightning Indexer cu_seqlens")
    if layout == "TND":
        if cu_q is None or cu_k is None:
            raise ValueError("TND Lightning Indexer requires cu_seqlens_q and cu_seqlens_k.")
        query_op = query.permute(1, 0, 2, 3).contiguous().reshape(-1, heads_q, head_dim)
        key_op = key.permute(1, 0, 2, 3).contiguous().reshape(-1, heads_k, head_dim)
        weights_op = weights.permute(1, 0, 2).contiguous().reshape(-1, heads_q)
        q_lengths = torch.diff(cu_q)
        k_lengths = torch.diff(cu_k)
        if max_seqlen_q is None:
            max_seqlen_q = int(q_lengths.max().item()) if q_lengths.numel() else 0
        if max_seqlen_k is None:
            max_seqlen_k = int(k_lengths.max().item()) if k_lengths.numel() else 0
        batch_for_metadata = int(cu_q.numel() - 1)
    else:
        query_op = query.permute(1, 0, 2, 3).contiguous()
        key_op = key.permute(1, 0, 2, 3).contiguous()
        weights_op = weights.permute(1, 0, 2).contiguous()
        max_seqlen_q = int(seq_q) if max_seqlen_q is None else int(max_seqlen_q)
        max_seqlen_k = int(seq_k) if max_seqlen_k is None else int(max_seqlen_k)
        batch_for_metadata = int(batch_size)

    if cmp_residual_k is None:
        residual_value = int(seq_k) % int(cmp_ratio)
        residual_values = (residual_value,) * int(batch_for_metadata)
        residual = torch.full(
            (batch_for_metadata,),
            residual_value,
            dtype=torch.int32,
            device=query.device,
        )
    else:
        residual = cmp_residual_k.to(device=query.device, dtype=torch.int32)
        if residual.dim() != 1 or residual.numel() != batch_for_metadata:
            raise ValueError("cmp_residual_k must contain one value per Lightning Indexer sample.")
        residual_values = tuple(int(value) for value in residual.detach().cpu().tolist())

    metadata = _metadata_cached(
        int(batch_for_metadata),
        int(max_seqlen_q),
        int(heads_q),
        int(head_dim),
        int(max_seqlen_k),
        int(heads_k),
        layout,
        None if cu_q is None else tuple(int(value) for value in cu_q.detach().cpu().tolist()),
        None if cu_k is None else tuple(int(value) for value in cu_k.detach().cpu().tolist()),
        int(topk),
        int(sparse_mode),
        int(cmp_ratio),
        residual_values,
        str(query.device),
    )
    indices, values = custom_ops.lightning_indexer(
        query_op,
        key_op,
        weights_op.float(),
        int(topk),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        cmp_residual_k=residual,
        metadata=metadata,
        layout_q=layout,
        layout_k=layout,
        mask_mode=int(sparse_mode),
        cmp_ratio=int(cmp_ratio),
        return_value=int(bool(return_values)),
    )
    if not return_values:
        values = None
    return indices, values


__all__ = [
    "get_npu_lightning_indexer_metadata",
    "npu_lightning_indexer",
]
