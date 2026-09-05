# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from typing import Optional

import torch

from ._smla_inputs import (
    align_deepseek_v4_cp_tensor,
    build_deepseek_v4_cp_smla_inputs,
)
from .ops.npu_sparse_flash_mla import (
    npu_sparse_flash_mla_from_smla_inputs,
    npu_sparse_flash_mla_with_indexer_loss_from_smla_inputs,
)


def _require_runtime_and_compressed_kv(
    compression_ratio,
    runtime_metadata,
    prepared_compressed_kv,
    cmp_sparse_indices,
):
    if compression_ratio not in (1, 4, 128):
        raise ValueError("compression_ratio must be one of 1, 4, or 128.")
    if runtime_metadata is None:
        raise ValueError("runtime_metadata is required.")
    if compression_ratio > 1 and prepared_compressed_kv is None:
        raise ValueError("prepared_compressed_kv is required when compression_ratio > 1.")
    if compression_ratio == 4 and cmp_sparse_indices is None:
        raise ValueError("cmp_sparse_indices is required for C4A.")


def run_deepseek_v4_cp_sparse_attention(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    compression_ratio: int,
    runtime_metadata,
    sinks: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    prepared_compressed_kv=None,
    cmp_sparse_indices: Optional[torch.Tensor] = None,
    cmp_sparse_indices_are_causal: bool = False,
    metadata: Optional[torch.Tensor] = None,
    layout_q: str = "TND",
    layout_kv: str = "TND",
    compacted_compressed_kv=None,
    compacted_block_starts=None,
):
    """Run the DeepSeek V4 CP path through SparseFlashMla.

    The caller prepares windowed ``ori_kv``, Attention-owned ``runtime_metadata``,
    and, when ``compression_ratio > 1``, ``prepared_compressed_kv``. C4A also
    requires ``cmp_sparse_indices``.
    """
    _require_runtime_and_compressed_kv(
        compression_ratio,
        runtime_metadata,
        prepared_compressed_kv,
        cmp_sparse_indices,
    )
    smla_inputs = build_deepseek_v4_cp_smla_inputs(
        q,
        ori_kv,
        compression_ratio,
        runtime_metadata=runtime_metadata,
        prepared_compressed_kv=prepared_compressed_kv,
        cmp_sparse_indices=cmp_sparse_indices,
        cmp_sparse_indices_are_causal=cmp_sparse_indices_are_causal,
        layout_q=layout_q,
        layout_kv=layout_kv,
        metadata=metadata,
        compacted_compressed_kv=compacted_compressed_kv,
        compacted_block_starts=compacted_block_starts,
    )
    return npu_sparse_flash_mla_from_smla_inputs(
        smla_inputs,
        softmax_scale=softmax_scale,
        sinks=sinks,
        cmp_ratio=compression_ratio,
        layout_q=layout_q,
        layout_kv=layout_kv,
    )


def run_deepseek_v4_cp_sparse_attention_with_indexer_loss(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    compression_ratio: int,
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weights: torch.Tensor,
    runtime_metadata,
    sinks: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    prepared_compressed_kv=None,
    cmp_sparse_indices: Optional[torch.Tensor] = None,
    cmp_sparse_indices_are_causal: bool = False,
    metadata: Optional[torch.Tensor] = None,
    layout_q: str = "TND",
    layout_kv: str = "TND",
    loss_tracker=None,
    loss_coeff: float = 1.0,
    compacted_compressed_kv=None,
    compacted_block_starts=None,
):
    """Run DeepSeek V4 CP C4A through SparseFlashMla with fused indexer loss."""
    if compression_ratio != 4:
        raise ValueError("DeepSeek V4 CP with indexer loss currently requires compression_ratio=4.")
    _require_runtime_and_compressed_kv(
        compression_ratio,
        runtime_metadata,
        prepared_compressed_kv,
        cmp_sparse_indices,
    )
    smla_inputs = build_deepseek_v4_cp_smla_inputs(
        q,
        ori_kv,
        compression_ratio,
        runtime_metadata=runtime_metadata,
        prepared_compressed_kv=prepared_compressed_kv,
        cmp_sparse_indices=cmp_sparse_indices,
        cmp_sparse_indices_are_causal=cmp_sparse_indices_are_causal,
        layout_q=layout_q,
        layout_kv=layout_kv,
        metadata=metadata,
        compacted_compressed_kv=compacted_compressed_kv,
        compacted_block_starts=compacted_block_starts,
    )
    if smla_inputs.alignment is None:
        raise ValueError("SMLA with indexer loss requires compressed-KV alignment metadata.")
    key_seq_dim = 0 if layout_kv == "TND" else 1
    key_index = align_deepseek_v4_cp_tensor(
        key_index,
        smla_inputs.alignment,
        key_seq_dim,
        tensor_name="key_index",
    )
    return npu_sparse_flash_mla_with_indexer_loss_from_smla_inputs(
        smla_inputs,
        query_index=query_index,
        key_index=key_index,
        weights=weights,
        softmax_scale=softmax_scale,
        sinks=sinks,
        cmp_ratio=compression_ratio,
        layout_q=layout_q,
        layout_kv=layout_kv,
        loss_tracker=loss_tracker,
        loss_coeff=loss_coeff,
    )
