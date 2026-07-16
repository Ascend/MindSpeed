# Copyright (c) 2024, Huawei Technologies Co., Ltd.
#
# DSA NPU fused operators (P1+P2+P3).
#
# Aligns with MindSpeed 0.16 dsa_fused.py's tightly-coupled design:
#   P2 (forward_with_scores) returns 4-tuple (topk_indices, q, k, weights)
#   P1 (fused_npu_sparse_flash_attention) returns (output, softmax_max, softmax_sum)
#   P3 (fused_compute_dsa_indexer_kl_loss) reuses P1+P2 precomputed results
#
# P1+P2+P3 are tightly coupled: P2's raw NPU op has no autograd (backward is
# handled by P3's LILossTrain), so they must be enabled together.
#
# Reference: MindSpeed 0.16 core_r0.16.0 dsa_fused.py

import torch
import torch_npu
from einops import rearrange

from megatron.core import mpu
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    compute_dsa_indexer_loss as megatron_compute_dsa_indexer_loss,
    unfused_dsa_fn as megatron_unfused_dsa_fn,
)

from mindspeed.core.transformer.experimental_attention_variant.utils import allgather_head_dim


def fused_lightning_indexer(
    q, k, weights, index_topk, actual_seq_qlen=None, actual_seq_klen=None, layout_query='BSND', layout_key='BSND'
):
    """P2: Fused NPU lightning indexer.

    Follows the MindSpeed 0.16 implementation and calls torch_npu's public
    npu_lightning_indexer API. Backward is handled by LILossTrain in P3.

    Args:
        q: Indexer Q after RoPE + Hadamard
           BSND: [seqlen, batch, index_n_heads, index_head_dim]
           TND:  [T, index_n_heads, index_head_dim]
        k: Indexer K after RoPE + Hadamard (head dim kept for NPU op)
           BSND: [seqlen, batch, 1, index_head_dim]
           TND:  [T, 1, index_head_dim]
        weights: Indexer weights
           BSND: [seqlen, batch, index_n_heads]
           TND:  [T, index_n_heads]
        index_topk: int, number of top-k indices to select
        actual_seq_qlen: TND cumulative seq lengths for query (int32)
        actual_seq_klen: TND cumulative seq lengths for key (int32)
        layout_query: 'BSND' or 'TND'
        layout_key: 'BSND' or 'TND'

    Returns:
        topk_indices: [b, sq, index_topk] (BSND) or [T, index_topk] (TND)
        topk_score: scores of selected indices
    """
    is_tnd = layout_query == 'TND'
    if is_tnd:
        q = q.to(torch.bfloat16).contiguous()
        k = k.to(torch.bfloat16).contiguous()
        weights = weights.to(torch.bfloat16).contiguous()
    else:
        q = rearrange(q, 's b h d -> b s h d').to(torch.bfloat16).contiguous()
        k = rearrange(k, 's b h d -> b s h d').to(torch.bfloat16).contiguous()
        weights = rearrange(weights, 's b d -> b s d').to(torch.bfloat16).contiguous()

    sparse_mode = 3
    sparse_indices, sparse_values = torch_npu.npu_lightning_indexer(
        q,
        k,
        weights,
        actual_seq_lengths_query=actual_seq_qlen,
        actual_seq_lengths_key=actual_seq_klen,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=index_topk,
        sparse_mode=sparse_mode,
        return_value=True,
    )

    sparse_indices = sparse_indices.squeeze(2)
    sparse_values = sparse_values.squeeze(2)
    return sparse_indices, sparse_values


def fused_npu_sparse_flash_attention(
    query, key, value, topk_indices, query_rope, key_rope, softmax_scale, packed_seq_params=None
):
    """P1: NPU fused sparse flash attention.

    Calls torch_npu.npu_sparse_flash_attention with return_softmax_lse=True
    to return softmax statistics for P3 KL loss computation.

    Args:
        query: [sq, b, np, qk_head_dim] query nope part, or kv_lora_rank in absorb mode
        key: [skv, b, np, qk_head_dim] key nope part, or kv_lora_rank in absorb mode
        value: [skv, b, np, v_head_dim]
        topk_indices: [b, sq, index_topk] (BSND) or [T, index_topk] (TND)
        query_rope: [sq, b, np, qk_pos_emb_head_dim]
        key_rope: [skv, b, np, qk_pos_emb_head_dim]
        softmax_scale: float
        packed_seq_params: PackedSeqParams for TND layout

    Returns:
        output: [sq, b, np, v_head_dim]
        softmax_max: softmax max statistics (for P3)
        softmax_sum: softmax sum statistics (for P3)
    """
    is_tnd = packed_seq_params is not None

    if is_tnd:
        actual_seq_qlen = packed_seq_params.cu_seqlens_q.to(torch.int32)
        actual_seq_kvlen = packed_seq_params.cu_seqlens_kv.to(torch.int32)
        layout = 'TND'
    else:
        query, key, value = [rearrange(x, 's b n d -> b s n d') for x in [query, key, value]]
        query_rope = rearrange(query_rope, 's b h d -> b s h d')
        key_rope = rearrange(key_rope, 's b h d -> b s h d')
        layout = 'BSND'

        batch_size = query.shape[0]
        seq_len = query.shape[1]
        actual_seq_qlen = torch.full((batch_size,), seq_len, dtype=torch.int32, device=query.device)
        actual_seq_kvlen = actual_seq_qlen

    if not is_tnd:
        topk_indices = topk_indices.unsqueeze(2)

    output, softmax_max, softmax_sum, *_ = torch_npu.npu_sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices=topk_indices.to(torch.int32),
        block_table=None,
        actual_seq_lengths_query=actual_seq_qlen,
        actual_seq_lengths_kv=actual_seq_kvlen,
        query_rope=query_rope,
        key_rope=key_rope,
        scale_value=softmax_scale,
        sparse_block_size=1,
        layout_query=layout,
        layout_kv=layout,
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=True,
    )

    if not is_tnd:
        output = rearrange(output, 'b s h d -> s b h d')

    return output, softmax_max, softmax_sum


def fused_sparse_lightning_indexer_kl_loss(
    query,
    key,
    query_index,
    key_index,
    weights,
    topk_indices,
    softmax_max,
    softmax_sum,
    scale_value=1,
    query_rope=None,
    key_rope=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout='BSND',
    sparse_mode=3,
    pre_tokens=65536,
    next_tokens=65536,
):
    """P3: NPU fused KL loss wrapper.

    Wraps LILossTrain.apply, following MindSpeed 0.16's verified P3 path.
    Handles TND/BSND layout differences and sb->bs transpose for BSND.

    Args:
        query: [sq, b, np, hn] main attention query (nope part)
        key: [skv, b, np, hn] main attention key (nope part)
        query_index: [sq, b, index_n_heads, index_head_dim] indexer Q from P2
        key_index: [sq, b, 1, index_head_dim] indexer K from P2
        weights: [sq, b, index_n_heads] indexer weights from P2
        topk_indices: [b, sq, index_topk]
        softmax_max: from P1
        softmax_sum: from P1
        scale_value: softmax scale
        query_rope: [sq, b, np, qk_pos_emb_head_dim] (optional)
        key_rope: [skv, b, np, qk_pos_emb_head_dim] (optional)
        actual_seq_qlen: TND cumulative seq lengths for query
        actual_seq_klen: TND cumulative seq lengths for key
        layout: 'BSND' or 'TND'

    Returns:
        loss: scalar tensor (scaled by 1/sq)
    """
    is_tnd = layout == 'TND'
    if is_tnd:
        # TND: tensors are [T, N, D], no B dimension to transpose
        sq = query.shape[0]
    else:
        # BSND: transpose sb -> bs for the NPU op
        query, key, query_index, key_index, weights = [
            x.transpose(0, 1) for x in [query, key, query_index, key_index, weights]
        ]
        if query_rope is not None:
            query_rope, key_rope = [x.transpose(0, 1) for x in [query_rope, key_rope]]
        sq = query.shape[1]

    if not is_tnd:
        topk_indices = topk_indices.unsqueeze(2)

    loss = LILossTrain.apply(
        query,
        key,
        query_index,
        key_index,
        weights,
        topk_indices,
        softmax_max,
        softmax_sum,
        scale_value,
        query_rope,
        key_rope,
        actual_seq_qlen,
        actual_seq_klen,
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens,
    )
    return loss / sq


class LILossTrain(torch.autograd.Function):
    """
    Custom autograd function for sparse lightning indexer KL loss.

    This mirrors MindSpeed 0.16's DSA P3 path: the forward calls torch_npu's
    public npu_sparse_lightning_indexer_grad_kl_loss API, and backward reuses
    the gradients returned by that fused operator.
    """

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        scale_value=1,
        query_rope=None,
        key_rope=None,
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout='BSND',
        sparse_mode=3,
        pre_tokens=65536,
        next_tokens=65536,
    ):
        d_query_index, d_key_index, d_weights, loss = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            query,
            key,
            query_index,
            key_index,
            weights,
            sparse_indices,
            softmax_max,
            softmax_sum,
            scale_value=scale_value,
            query_rope=query_rope,
            key_rope=key_rope,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout=layout,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
        )

        ctx.save_for_backward(d_query_index, d_key_index, d_weights)
        return loss[0]

    @staticmethod
    def backward(ctx, *grad_output):
        d_query_index, d_key_index, d_weights = ctx.saved_tensors
        grad_scale = grad_output[0]
        if torch.ne(grad_scale, torch.tensor(1.0, device=grad_scale.device)):
            d_query_index = d_query_index * grad_scale
            d_key_index = d_key_index * grad_scale
            d_weights = d_weights * grad_scale

        res_list = [None] * 12
        return None, None, d_query_index, d_key_index, d_weights, *res_list


def fused_compute_dsa_indexer_kl_loss(
    topk_indices,
    query,
    key,
    softmax_scale,
    loss_coeff,
    q_pos_emb,
    k_pos_emb,
    query_index,
    key_index,
    weights,
    softmax_max,
    softmax_sum,
    packed_seq_params,
    tensor_model_parallel_size=1,
):
    """P3: Orchestrate NPU fused KL loss computation.

    Reuses precomputed results from P1 (softmax_max, softmax_sum) and P2
    (topk_indices, query_index, key_index, weights). Handles TP allgather
    for query and q_pos_emb.

    Args:
        topk_indices: [b, sq, index_topk] from P2
        query: [sq, b, np, qk_head_dim] main attention query nope part, or kv_lora_rank in absorb mode
        key: [skv, b, np, qk_head_dim] main attention key nope part, or kv_lora_rank in absorb mode
        softmax_scale: float
        loss_coeff: float, loss coefficient
        q_pos_emb: [sq, b, np, qk_pos_emb_head_dim] query rope part
        k_pos_emb: [skv, b, np, qk_pos_emb_head_dim] key rope part
        query_index: [sq, b, index_n_heads, index_head_dim] indexer Q from P2
        key_index: [sq, b, 1, index_head_dim] indexer K from P2
        weights: [sq, b, index_n_heads] indexer weights from P2
        softmax_max: from P1
        softmax_sum: from P1
        packed_seq_params: PackedSeqParams for TND
        tensor_model_parallel_size: int

    Returns:
        indexer_loss: scalar tensor
    """
    use_tnd = packed_seq_params is not None
    actual_seq_qlen = packed_seq_params.cu_seqlens_q.to(torch.int32) if use_tnd else None
    actual_seq_klen = packed_seq_params.cu_seqlens_kv.to(torch.int32) if use_tnd else None
    layout = 'TND' if use_tnd else 'BSND'

    if tensor_model_parallel_size > 1:
        tp_group = mpu.get_tensor_model_parallel_group()
        total_query = allgather_head_dim(query, tensor_model_parallel_size, tp_group, layout=layout)
        total_query_rope = allgather_head_dim(q_pos_emb, tensor_model_parallel_size, tp_group, layout=layout)
        softmax_max = gather_from_tensor_model_parallel_region(softmax_max)
        softmax_sum = gather_from_tensor_model_parallel_region(softmax_sum)
    else:
        total_query = query
        total_query_rope = q_pos_emb

    from mindspeed.args_utils import get_full_args as get_args

    args = get_args()
    if args.context_parallel_size > 1 and args.context_parallel_algo == 'kvallgather_cp_algo':
        from mindspeed.core.transformer.experimental_attention_variant.dsa_kvallgather_context_parallel import (
            fused_sparse_lightning_indexer_kl_loss_kvallgather,
        )

        cp_group = mpu.get_context_parallel_group()
        loss = fused_sparse_lightning_indexer_kl_loss_kvallgather(
            total_query,
            key,
            query_index,
            key_index,
            weights,
            topk_indices,
            softmax_max,
            softmax_sum,
            scale_value=softmax_scale,
            query_rope=total_query_rope,
            key_rope=k_pos_emb,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout=layout,
            cp_group=cp_group,
        )
    else:
        loss = fused_sparse_lightning_indexer_kl_loss(
            total_query,
            key,
            query_index,
            key_index,
            weights,
            topk_indices,
            softmax_max,
            softmax_sum,
            scale_value=softmax_scale,
            query_rope=total_query_rope,
            key_rope=k_pos_emb,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout=layout,
        )

    indexer_loss = loss * loss_coeff
    return indexer_loss


# Stores the original DSAIndexer.forward_with_scores for fallback
_original_DSAIndexer_forward_with_scores = None


def forward_with_scores(self, x, qr, mask=None, packed_seq_params=None, use_fused_lightning_indexer=False):
    """P2: Replace DSAIndexer.forward_with_scores with NPU fused path.

    When use_fused_lightning_indexer=True:
        Uses npu_lightning_indexer (raw op) and returns 4-tuple
        (topk_indices, q, k, weights) for P3 KL loss reuse.
    When use_fused_lightning_indexer=False:
        Falls back to native Megatron forward_with_scores, returns 2-tuple
        (index_scores, topk_indices).

    Supports TND (packed_seq) and BSND layouts.
    """
    if not use_fused_lightning_indexer:
        if _original_DSAIndexer_forward_with_scores is None:
            raise RuntimeError("DSA indexer fallback was used before its original method was registered.")
        # The feature registration assigns this dynamic monkey-patch callback before use.
        return _original_DSAIndexer_forward_with_scores(  # pylint: disable=not-callable
            self, x, qr, mask, packed_seq_params
        )

    from mindspeed.args_utils import get_full_args as get_args

    args = get_args()

    use_tnd = packed_seq_params is not None

    # Prepare RoPE params
    rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(None, None, x, self.config, packed_seq_params)
    if self.config.rope_type == "rope":
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=use_tnd)
        mscale = 1.0
    else:
        rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=use_tnd)

    # Gather inputs if SP is enabled
    if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
        x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
        qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

    # Get sequence length and batch size
    if use_tnd:
        seqlen, bsz, hsz = x.size()
        x = x.reshape(seqlen * bsz, hsz)
        seqlen = seqlen * bsz
        bsz = 1
    else:
        seqlen, bsz, _ = x.size()

    # q linear and apply rope to q
    q, _ = self.linear_wq_b(qr)
    if use_tnd:
        q = q.reshape(seqlen, self.index_n_heads, self.index_head_dim)
        nope_dim = self.index_head_dim - self.qk_pos_emb_head_dim
        q_nope, q_pe = torch.split(q, [nope_dim, self.qk_pos_emb_head_dim], dim=-1)
        from megatron.core.models.common.embeddings import apply_rotary_pos_emb

        cu_seqlens = packed_seq_params.cu_seqlens_q
        if getattr(args, 'apply_rope_in_complex', False):
            from mindspeed.core.transformer.experimental_attention_variant.dsa_rope import apply_rope_in_complex

            q_pe = apply_rope_in_complex(q_pe, rotary_pos_emb, mscale=mscale)
        else:
            q_pe = apply_rotary_pos_emb(q_pe, rotary_pos_emb, config=self.config, cu_seqlens=cu_seqlens, mscale=mscale)
        q = torch.cat([q_nope, q_pe], dim=-1)
    else:
        q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)
        q = self._apply_rope(q, rotary_pos_emb, mscale)

    # k linear and apply rope to k
    k, _ = self.linear_wk(x)
    k = self.k_norm(k)
    if use_tnd:
        k = k.reshape(seqlen, 1, self.index_head_dim)
        nope_dim = self.index_head_dim - self.qk_pos_emb_head_dim
        k_nope, k_pe = torch.split(k, [nope_dim, self.qk_pos_emb_head_dim], dim=-1)
        cu_seqlens = packed_seq_params.cu_seqlens_kv
        if getattr(args, 'apply_rope_in_complex', False):
            from mindspeed.core.transformer.experimental_attention_variant.dsa_rope import apply_rope_in_complex

            k_pe = apply_rope_in_complex(k_pe, rotary_pos_emb, mscale=mscale)
        else:
            k_pe = apply_rotary_pos_emb(k_pe, rotary_pos_emb, config=self.config, cu_seqlens=cu_seqlens, mscale=mscale)
        k = torch.cat([k_nope, k_pe], dim=-1)
    else:
        # Keep head dim (size 1) for the NPU op
        k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
        k = self._apply_rope(k, rotary_pos_emb, mscale)

    # Rotate activation (Hadamard)
    from megatron.core.transformer.experimental_attention_variant.dsa import rotate_activation

    q = rotate_activation(q)
    k = rotate_activation(k)

    # Compute weights
    weights, _ = self.linear_weights_proj(x)
    weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale

    actual_seq_qlen = packed_seq_params.cu_seqlens_q.to(torch.int32) if use_tnd else None
    actual_seq_klen = packed_seq_params.cu_seqlens_kv.to(torch.int32) if use_tnd else None
    layout = 'TND' if use_tnd else 'BSND'

    # CP-aware lightning indexer
    if args.context_parallel_size > 1 and args.context_parallel_algo == 'kvallgather_cp_algo':
        from mindspeed.core.transformer.experimental_attention_variant.dsa_kvallgather_context_parallel import (
            fused_lightning_indexer_kvallgather,
        )

        cp_group = self.pg_collection.cp
        topk_indices, _ = fused_lightning_indexer_kvallgather(
            q,
            k,
            weights,
            self.index_topk,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout_query=layout,
            layout_key=layout,
            cp_group=cp_group,
        )
    else:
        topk_indices, _ = fused_lightning_indexer(
            q,
            k,
            weights,
            self.index_topk,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout_query=layout,
            layout_key=layout,
        )

    return topk_indices, q, k, weights


def fused_dsa_attn_forward(
    self, query, key, value, attention_mask, x, qr, attn_mask_type=None, attention_bias=None, packed_seq_params=None
):
    """P1: Replace DSAttention.forward with NPU fused path.

    Integrates P2 (fused lightning indexer) and P3 (fused KL loss) in a
    tightly-coupled design:
      1. P2: forward_with_scores(use_fused_lightning_indexer=True)
         -> (topk_indices, query_index, key_index, weights)
      2. P1: fused_npu_sparse_flash_attention
         -> (output, softmax_max, softmax_sum)
      3. P3: fused_compute_dsa_indexer_kl_loss reuses P1+P2 precomputed results

    Training and inference share the same P2 path; P3 only runs in training.
    When use_fused_lightning_indexer is off, falls back to native indexer path.
    """
    from mindspeed.args_utils import get_full_args as get_args

    args = get_args()

    use_tnd = packed_seq_params is not None
    if use_tnd:
        sq, np, hn = query.size()
        b = 1
        skv = key.size(0)
    else:
        sq, b, np, hn = query.size()
        skv = key.size(0)

    # Detach x and qr to prevent gradients from flowing back to main model
    x = x.detach()
    qr = qr.detach()

    # P2: Get topk indices (and q/k/weights for P3 reuse when fused)
    if self.config.use_fused_lightning_indexer:
        topk_indices, query_index, key_index, weights = self.indexer.forward_with_scores(
            x,
            qr,
            mask=None,
            packed_seq_params=packed_seq_params,
            use_fused_lightning_indexer=True,
        )
        index_scores = None
    else:
        # Native fallback: build float mask and call original forward_with_scores
        if attn_mask_type is not None:
            if attn_mask_type != AttnMaskType.causal:
                raise RuntimeError(f"Only causal mask is supported for now, but got attn_mask_type={attn_mask_type}")
            float_mask = torch.triu(
                torch.full((sq, skv), float('-inf'), dtype=torch.float32, device=x.device),
                diagonal=1,
            )
        else:
            if not use_tnd and attention_mask.shape != (b, 1, sq, skv):
                raise ValueError(f"Expected attention_mask shape {(b, 1, sq, skv)}, but got {attention_mask.shape}")
            mask = attention_mask.squeeze()
            float_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(mask, float('-inf'))

        index_scores, topk_indices = self.indexer.forward_with_scores(
            x,
            qr,
            mask=float_mask,
            packed_seq_params=packed_seq_params,
        )
        query_index = key_index = weights = None

    # Split main attention query/key into nope and rope parts for sparse attention.
    # Megatron 0.17 MLA passes qk_head_dim + rope; 0.16-style DSA absorption passes
    # kv_lora_rank + rope after absorbing K up-projection into Q.
    rope_dim = self.config.qk_pos_emb_head_dim
    split_candidates = [
        ("qk_head_dim", self.config.qk_head_dim),
        ("kv_lora_rank", self.config.kv_lora_rank),
    ]
    qk_split_sizes = None
    for dim_name, nope_dim in split_candidates:
        expected_qk_dim = nope_dim + rope_dim
        if query.size(-1) == expected_qk_dim and key.size(-1) == expected_qk_dim:
            qk_split_sizes = [nope_dim, rope_dim]
            break
    if qk_split_sizes is None:
        valid_shapes = ", ".join(
            f"{dim_name} + qk_pos_emb_head_dim = {nope_dim + rope_dim}" for dim_name, nope_dim in split_candidates
        )
        raise RuntimeError(
            "DSA fused sparse attention expects query/key last dim to equal one of "
            f"{valid_shapes}, "
            f"but got query={query.size(-1)}, key={key.size(-1)}."
        )
    query_nope, query_rope = torch.split(query, qk_split_sizes, dim=-1)
    key_nope, key_rope = torch.split(key, qk_split_sizes, dim=-1)

    # P1: Sparse attention
    if self.config.use_fused_sparse_flash_attention:
        if args.context_parallel_size > 1 and args.context_parallel_algo == 'kvallgather_cp_algo':
            from mindspeed.core.transformer.experimental_attention_variant.dsa_kvallgather_context_parallel import (
                fused_npu_sparse_flash_attention_kvallgather,
            )

            cp_group = mpu.get_context_parallel_group()
            cp_stream = torch.npu.Stream(device=torch.npu.current_device())
            output, softmax_max, softmax_sum = fused_npu_sparse_flash_attention_kvallgather(
                query_nope,
                key_nope,
                value,
                topk_indices,
                query_rope,
                key_rope,
                self.softmax_scale,
                cp_group=cp_group,
                cp_stream=cp_stream,
                layout='TND' if use_tnd else 'BSND',
                packed_seq_params=packed_seq_params,
            )
        else:
            output, softmax_max, softmax_sum = fused_npu_sparse_flash_attention(
                query_nope,
                key_nope,
                value,
                topk_indices,
                query_rope,
                key_rope,
                self.softmax_scale,
                packed_seq_params=packed_seq_params,
            )
    else:
        if use_tnd:
            raise RuntimeError("unfused_dsa_fn does not support TND format. Use --use-fused-sparse-flash-attention.")
        output = megatron_unfused_dsa_fn(query_nope, key_nope, value, topk_indices, self.softmax_scale)
        softmax_max = None
        softmax_sum = None

    # P3: Attach indexer loss (training only)
    if self.training and torch.is_grad_enabled():
        indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0)

        if self.config.use_fused_lightning_indexer_kl_loss:
            indexer_loss = fused_compute_dsa_indexer_kl_loss(
                topk_indices,
                query_nope.detach(),
                key_nope.detach(),
                self.softmax_scale,
                indexer_loss_coeff,
                query_rope.detach(),
                key_rope.detach(),
                query_index,
                key_index,
                weights,
                softmax_max.detach() if softmax_max is not None else None,
                softmax_sum.detach() if softmax_sum is not None else None,
                packed_seq_params,
                tensor_model_parallel_size=self.config.tensor_model_parallel_size,
            )
        else:
            # Native KL loss fallback (requires index_scores from native indexer)
            query_full = torch.cat([query_nope, query_rope], dim=-1)
            key_full = torch.cat([key_nope, key_rope], dim=-1)
            indexer_loss = megatron_compute_dsa_indexer_loss(
                index_scores,
                topk_indices,
                query_full.detach(),
                key_full.detach(),
                self.softmax_scale,
                indexer_loss_coeff,
                getattr(self.config, "dsa_indexer_use_sparse_loss", False),
                self.indexer.pg_collection,
            )

        if indexer_loss_coeff > 0:
            mtp_num_layers = getattr(self.config, 'mtp_num_layers', None)
            num_layers = self.config.num_layers + mtp_num_layers if mtp_num_layers else self.config.num_layers
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=indexer_loss,
                layer_number=self.layer_number,
                num_layers=num_layers,
            )
        output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

    return output
