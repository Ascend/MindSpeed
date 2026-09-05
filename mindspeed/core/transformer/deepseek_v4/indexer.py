# Copyright (c) 2025 NVIDIA CORPORATION.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from dataclasses import dataclass
from typing import Union

import torch
from einops import rearrange

from megatron.core.transformer import MegatronModule, ModuleSpec, build_module
from megatron.core.transformer.identity_op import IdentityOp
from megatron.training import get_args

from .compressor import get_compressor_spec
from .deepseek_utils import (
    apply_rotary_emb,
    max_seqlen_from_cu_seqlens,
    rotate_activation,
)
from mindspeed.core.context_parallel.deepseek_v4_context_parallel._utils import normalize_cu_seqlens
from .linear import LinearNoTP
from mindspeed.core.context_parallel.deepseek_v4_context_parallel.ops.npu_lightning_indexer import (
    npu_lightning_indexer,
)


@dataclass
class DSAIndexerSubmodules:
    wq_b: Union[ModuleSpec, type] = None
    weights_proj: Union[ModuleSpec, type] = None
    compressor: Union[ModuleSpec, type] = None


def get_dsa_indexer_spec(enable_dsa_indexer, compressor=None):
    if not enable_dsa_indexer:
        return IdentityOp
    return ModuleSpec(
        module=DSAIndexer,
        submodules=DSAIndexerSubmodules(
            wq_b=LinearNoTP,
            weights_proj=LinearNoTP,
            compressor=get_compressor_spec() if compressor else IdentityOp,
        ),
    )


class DSAIndexer(MegatronModule):
    """DeepSeek-V4 learned query/key scoring module used by ds-cp attention.

    Query/weight projection lives here. Dense top-k and fused Lightning Indexer
    orchestration live in ``cp_indexer``.
    """

    def __init__(self, config, submodules: DSAIndexerSubmodules, layer_number: int):
        super().__init__(config=config)
        args = get_args()
        if not bool(args.kv_compress):
            raise RuntimeError("DeepSeek-V4 CP DSAIndexer requires kv_compress.")
        self.dim = int(args.hidden_size)
        self.n_heads = int(args.index_n_heads)
        self.head_dim = int(args.index_head_dim)
        self.rope_head_dim = int(args.qk_pos_emb_head_dim)
        self.index_topk = int(args.index_topk)
        self.q_lora_rank = int(args.q_lora_rank)
        self.use_fused_lightning_indexer = bool(getattr(args, "use_fused_lightning_indexer", False))
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = int(args.compress_ratios[layer_number - 1])

        self.wq_b = build_module(
            submodules.wq_b,
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
        )
        self.kv_compressor = build_module(
            submodules.compressor,
            config=self.config,
            compress_ratio=self.compress_ratio,
            head_dim=self.head_dim,
            rotate=True,
        )
        self.weights_proj = build_module(
            submodules.weights_proj,
            self.dim,
            self.n_heads,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
        )

    def forward_query_weights(self, hidden_states, query_lora, freqs_cis, q_rope_preapplied=False):
        q = rearrange(self.wq_b(query_lora), "s b (h d) -> s b h d", d=self.head_dim)
        q = q.transpose(0, 1).contiguous()
        if not q_rope_preapplied:
            q[..., -self.rope_head_dim :] = apply_rotary_emb(q[..., -self.rope_head_dim :], freqs_cis)
        q = rotate_activation(q.transpose(0, 1).contiguous())
        source = hidden_states
        if source.shape[0] != q.shape[0]:
            source = source[-q.shape[0] :]
        weights = self.weights_proj(source) * self.n_heads**-0.5 * self.softmax_scale
        return q, weights

    def forward_with_scores_compress(
        self,
        q,
        k,
        weights,
        packed_seq_params,
        index_topk,
        offset,
        compress_ratio=4,
        cmp_residual_k=None,
        return_scores=True,
    ):
        """Run the fused Lightning Indexer on model-layout tensors.

        The CP right-alignment path invokes this method on a causal prefix
        segment.  ``offset`` converts the segment-local key ordinal back to
        the global compressed-block ordinal.
        """
        if not self.use_fused_lightning_indexer:
            raise RuntimeError("forward_with_scores_compress requires use_fused_lightning_indexer.")
        if q.dim() != 4 or k.dim() != 4 or weights.dim() != 3:
            raise ValueError("fused Indexer expects q/k/weights in [S, B, N, D]/[S, B, N] layout.")

        layout = "TND" if packed_seq_params is not None else "BSND"
        cu_q = cu_k = None
        max_q = max_k = None
        if packed_seq_params is not None:
            cu_q = getattr(packed_seq_params, "cu_seqlens_q", None)
            cu_k = getattr(packed_seq_params, "cu_seqlens_kv", None)
            if cu_q is None or cu_k is None:
                raise ValueError("packed_seq_params must provide cu_seqlens_q and cu_seqlens_kv.")
            cu_q = normalize_cu_seqlens(cu_q, q.device)
            cu_k = normalize_cu_seqlens(cu_k, q.device)
            max_q = max_seqlen_from_cu_seqlens(cu_q)
            max_k = max_seqlen_from_cu_seqlens(cu_k)

        indices, scores = npu_lightning_indexer(
            q,
            k,
            weights,
            int(index_topk),
            layout=layout,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            cmp_residual_k=cmp_residual_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            sparse_mode=3,
            cmp_ratio=int(compress_ratio),
            return_values=return_scores,
        )
        if int(offset) != 0:
            indices = torch.where(
                indices >= 0,
                indices + int(offset),
                indices,
            )
        return indices, scores if return_scores else None


__all__ = ["DSAIndexer", "DSAIndexerSubmodules", "get_dsa_indexer_spec"]
