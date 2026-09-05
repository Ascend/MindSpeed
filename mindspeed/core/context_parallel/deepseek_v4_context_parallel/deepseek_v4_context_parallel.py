# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""DeepSeek-V4 Context Parallel SMLA dispatch.

The caller prepares windowed ``ori_kv``, Attention-owned runtime metadata, and,
when ``compression_ratio > 1``, ``prepared_compressed_kv``. C4A also requires
``cmp_sparse_indices``. This module builds SMLA inputs and invokes SparseFlashMla.
"""

from typing import Optional

import torch

from ._types import (
    DeepSeekV4CPCompressedKV,
)
from .deepseek_v4_attention import (
    run_deepseek_v4_cp_sparse_attention,
    run_deepseek_v4_cp_sparse_attention_with_indexer_loss,
)


class DeepSeekV4CPContextParallel:
    """SMLA dispatch for DeepSeek-V4 context-parallel attention.

    Compression, window exchange, sparse-index selection, and runtime metadata
    belong to the caller. This class only constructs SMLA inputs and runs
    SparseFlashMla.
    """

    def __init__(
        self,
        compression_ratio: int,
        layout_q: str = "TND",
        layout_kv: str = "TND",
    ):
        if compression_ratio not in (1, 4, 128):
            raise ValueError(f"compression_ratio must be 1, 4, or 128, got {compression_ratio}")
        if layout_q not in ("TND", "BSND"):
            raise ValueError(f"layout_q must be TND or BSND, got {layout_q}")
        if layout_kv not in ("TND", "BSND"):
            raise ValueError(f"layout_kv must be TND or BSND, got {layout_kv}")
        if layout_q != layout_kv:
            raise ValueError(f"layout_q and layout_kv must match, got {layout_q} and {layout_kv}")

        self.compression_ratio = compression_ratio
        self.layout_q = layout_q
        self.layout_kv = layout_kv

    def forward(
        self,
        q: torch.Tensor,
        ori_kv: torch.Tensor,
        runtime_metadata,
        prepared_compressed_kv: Optional[DeepSeekV4CPCompressedKV] = None,
        sinks: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        cmp_sparse_indices: Optional[torch.Tensor] = None,
        cmp_sparse_indices_are_causal: bool = False,
        metadata: Optional[torch.Tensor] = None,
        query_index: Optional[torch.Tensor] = None,
        key_index: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        loss_tracker=None,
        loss_coeff: float = 1.0,
        compacted_compressed_kv=None,
        compacted_block_starts=None,
    ) -> torch.Tensor:
        """Apply DeepSeek-V4 context parallel attention."""
        use_indexer_loss = query_index is not None or key_index is not None or weights is not None
        if use_indexer_loss:
            return run_deepseek_v4_cp_sparse_attention_with_indexer_loss(
                q=q,
                ori_kv=ori_kv,
                compression_ratio=self.compression_ratio,
                query_index=query_index,
                key_index=key_index,
                weights=weights,
                runtime_metadata=runtime_metadata,
                sinks=sinks,
                softmax_scale=softmax_scale,
                prepared_compressed_kv=prepared_compressed_kv,
                cmp_sparse_indices=cmp_sparse_indices,
                cmp_sparse_indices_are_causal=cmp_sparse_indices_are_causal,
                metadata=metadata,
                layout_q=self.layout_q,
                layout_kv=self.layout_kv,
                loss_tracker=loss_tracker,
                loss_coeff=loss_coeff,
                compacted_compressed_kv=compacted_compressed_kv,
                compacted_block_starts=compacted_block_starts,
            )
        return run_deepseek_v4_cp_sparse_attention(
            q=q,
            ori_kv=ori_kv,
            compression_ratio=self.compression_ratio,
            runtime_metadata=runtime_metadata,
            sinks=sinks,
            softmax_scale=softmax_scale,
            prepared_compressed_kv=prepared_compressed_kv,
            cmp_sparse_indices=cmp_sparse_indices,
            cmp_sparse_indices_are_causal=cmp_sparse_indices_are_causal,
            metadata=metadata,
            layout_q=self.layout_q,
            layout_kv=self.layout_kv,
            compacted_compressed_kv=compacted_compressed_kv,
            compacted_block_starts=compacted_block_starts,
        )


__all__ = [
    "DeepSeekV4CPContextParallel",
]
