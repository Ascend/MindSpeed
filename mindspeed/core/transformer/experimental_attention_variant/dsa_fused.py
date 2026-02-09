# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025; Huawei Technologies Co., Ltd.  All rights reserved.

from typing import Optional, Tuple

import torch
import torch_npu

try:
    from einops import rearrange

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

from megatron.training import get_args
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossLoggingHelper,
    unfused_dsa_fn,
    DSAIndexerLossAutoScaler,
    rotate_activation,
    compute_dsa_indexer_loss
)
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType

from mindspeed.core.transformer.experimental_attention_variant.dsa_kvallgather_context_parallel import (
    fused_lightning_indexer_kvallgather,
    fused_sparse_lightning_indexer_kl_loss_kvallgather,
    fused_npu_sparse_flash_attention_kvallgather)

from mindspeed.core.transformer.experimental_attention_variant.utils import allgather_head_dim


# compute with fused and naive dsa
def fused_dsa_attn_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    x: torch.Tensor,
    qr: torch.Tensor,
    attention_mask: torch.Tensor,
    attn_mask_type: AttnMaskType = None,
    attention_bias: torch.Tensor = None,
    packed_seq_params: PackedSeqParams = None,
):
    """
    Forward pass for Sparse Attention.

    Args:
        query: Query tensor [sq, b, np, hn].
        key: Key tensor [skv, b, np, hn].
        value: Value tensor [skv, b, np, hnv].
        x: Original hidden states [sq, b, hidden_size].
        qr: Low-rank query representation [sq, b, q_lora_rank].
        attention_mask: Attention mask tensor [b, 1, sq, sk].
        attn_mask_type: Type of attention mask.
        attention_bias: Optional attention bias.
        packed_seq_params: Packed sequence parameters.

    Returns:
        output: Output tensor [sq, b, hidden_size]
    """

    args = get_args()

    sq, b, np, hn = query.size()
    skv = key.size(0)
    hnv = value.size(3)

    # Detach x and qr to prevent gradients of indexer from flowing back to the main model.
    x = x.detach()
    qr = qr.detach()

    # ===================================
    # Get index scores and top-k indices
    # ===================================
    if self.config.use_fused_lightning_indexer:
        topk_indices, query_index, key_index, weights = self.indexer.forward_with_scores(
            x, qr, mask=None, packed_seq_params=packed_seq_params, use_fused_lightning_indexer=True
        )

    else:
        # Get a FP32 mask with -inf for masked positions.
        if attn_mask_type is not None:
            assert attn_mask_type == AttnMaskType.causal, 'Only causal mask is supported for now'
            # Generate upper triangular mask with -inf above diagonal, 0 elsewhere
            # torch.triu with diagonal=1 creates upper triangular matrix (excluding main diagonal)
            # float_mask [sq, skv]
            float_mask = torch.triu(
                torch.full((sq, skv), float('-inf'), dtype=torch.float32, device=x.device),
                diagonal=1,
            )
        else:
            assert attention_mask.shape == (b, 1, sq, skv), 'attention_mask shape mismatch'
            # [b, 1, sq, skv] -> [b, sq, skv]
            mask = attention_mask.squeeze()
            # float_mask [b, sq, skv]
            float_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(
                mask, float('-inf')
            )

        index_scores, topk_indices = self.indexer.forward_with_scores(
            x, qr, mask=float_mask, packed_seq_params=packed_seq_params
        )

    query, query_rope = torch.split(query, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1)
    key, key_rope = torch.split(key, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1)


    if self.config.use_fused_sparse_flash_attention:
        if args.context_parallel_size > 1 and args.context_parallel_algo == 'kvallgather_cp_algo':
            cp_group = mpu.get_context_parallel_group()
            cp_stream = torch.npu.Stream(device=torch.npu.current_device())

            output, softmax_max, softmax_sum = fused_npu_sparse_flash_attention_kvallgather(
            query,
            key,
            value,
            topk_indices,
            query_rope,
            key_rope,
            self.softmax_scale,
            cp_group = cp_group,
            cp_stream = cp_stream
            )
        else:
            output, softmax_max, softmax_sum = fused_npu_sparse_flash_attention(
            query,
            key,
            value,
            topk_indices,
            query_rope,
            key_rope,
            self.softmax_scale)
    else:
        output = unfused_dsa_fn(query, key, value, topk_indices, self.softmax_scale)

    # ===================================
    # Attach indexer loss
    # ===================================
    if self.training and torch.is_grad_enabled():
        # Compute KL divergence loss between indexer scores and true attention scores
        indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0)
        if self.config.use_fused_lightning_indexer_kl_loss:
            indexer_loss = fused_compute_dsa_indexer_kl_loss(
                topk_indices,
                query.detach(),
                key.detach(),
                self.softmax_scale,
                indexer_loss_coeff,
                query_rope.detach(),
                key_rope.detach(),
                query_index,
                key_index,
                weights,
                softmax_max.detach(),
                softmax_sum.detach(),
                packed_seq_params,
                tensor_model_parallel_size=self.config.tensor_model_parallel_size
            )
        else:
            # For absorb mode, query and key are list format, they need to be concatenated for dsa indexer
            query = torch.cat([query, query_rope], dim=-1)
            key = torch.cat([key, key_rope], dim=-1)
            indexer_loss = compute_dsa_indexer_loss(
                index_scores,
                topk_indices,
                query.detach(),
                key.detach(),
                self.softmax_scale,
                indexer_loss_coeff,
                getattr(self.config, "dsa_indexer_use_sparse_loss", False),
                self.indexer.pg_collection,
            )
        # Save indexer loss for logging
        if indexer_loss_coeff > 0:
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=indexer_loss,
                layer_number=self.layer_number,
                num_layers=self.config.num_layers,
            )
        # Attach loss to output
        output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

    return output


def forward_with_scores(
    self,
    x: torch.Tensor,
    qr: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    use_fused_lightning_indexer: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for DSA Indexer that returns both index scores and top-k indices.

    This is used when KL loss is enabled to compare indexer scores with true attention scores.

    Args:
        x: hidden states [seqlen, batch, hidden_size].
        qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
        mask: Attention mask [batch, seqlen, seqlen].
        packed_seq_params: Packed sequence parameters for variable length sequences.

    Returns:
        index_scores: Index scores [batch, seqlen, seqlen].
        topk_indices: Top-k indices [batch, seqlen, index_topk].
    """

    args = get_args()

    assert packed_seq_params is None, "Packed sequence is not supported for DSAttention"

    # =========================================
    # Prepare RoPE params
    # =========================================
    rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
        None, None, x, self.config, packed_seq_params
    )
    if self.config.rope_type == "rope":
        rotary_pos_emb = self.rotary_pos_emb(
            rotary_seq_len, packed_seq_params=packed_seq_params
        )
        mscale = 1.0
    else:
        rotary_pos_emb, mscale = self.rotary_pos_emb(
            rotary_seq_len, packed_seq_params=packed_seq_params
        )

    # =========================================
    # Gather inputs if sp is enabled
    # =========================================
    if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
        x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
        qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

    # =========================================
    # Get sequence length and batch size
    # =========================================
    seqlen, bsz, _ = x.size()

    # =========================================
    # q linear and apply rope to q
    # =========================================
    # [seqlen, batch, q_lora_rank] -> [seqlen, batch, index_n_heads * index_head_dim]
    q, _ = self.linear_wq_b(qr)
    # [seqlen, batch, index_n_heads * index_head_dim]
    #   -> [seqlen, batch, index_n_heads, index_head_dim]
    q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)
    q = self._apply_rope(q, rotary_pos_emb, mscale)

    # =========================================
    # k linear and apply rope to k
    # =========================================
    # [seqlen, batch, hidden_size] -> [seqlen, batch, index_head_dim]
    k, _ = self.linear_wk(x)
    k = self.k_norm(k)
    # [seqlen, batch, index_head_dim] -> [seqlen, batch, 1, index_head_dim]
    k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
    k = self._apply_rope(k, rotary_pos_emb, mscale)
    if not use_fused_lightning_indexer:
        k = k.reshape(seqlen, bsz, self.index_head_dim)
    # =========================================
    # Rotate activation
    # =========================================
    q = rotate_activation(q)
    k = rotate_activation(k)

    # =========================================
    # Compute index scores
    # =========================================
    # [seqlen, batch, hidden_size] -> [seqlen, batch, index_n_heads]
    weights, _ = self.linear_weights_proj(x)
    weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale
    # [batch, seqlen, seqlen]
    if use_fused_lightning_indexer:
        if args.context_parallel_size > 1 and args.context_parallel_algo == 'kvallgather_cp_algo':
            cp_group = self.pg_collection.cp
            cp_stream = torch.npu.Stream(device=torch.npu.current_device())

            topk_indices, _ = fused_lightning_indexer_kvallgather(
                q,
                k,
                weights,
                self.index_topk,
                actual_seq_qlen=None if packed_seq_params is None else packed_seq_params.cu_seqlens_q,
                actual_seq_klen=None if packed_seq_params is None else packed_seq_params.cu_seqlens_kv,
                layout_query='BSND',
                layout_key='BSND',
                cp_group=cp_group,
                cp_stream=cp_stream
            )
        else:
            topk_indices, _ = fused_lightning_indexer(
                q,
                k,
                weights,
                self.index_topk,
                actual_seq_qlen=None if packed_seq_params is None else packed_seq_params.cu_seqlens_q,
                actual_seq_klen=None if packed_seq_params is None else packed_seq_params.cu_seqlens_kv,
                layout_query='BSND',
                layout_key='BSND',
            )
        return topk_indices, q, k, weights
    else:
        index_scores = self._compute_index_scores(q, weights, k)
        if mask is not None:
            assert mask.dtype == index_scores.dtype, "Mask dtype must match index scores dtype"
            index_scores = index_scores + mask

        # =========================================
        # Select top-k indices
        # =========================================
        topk_k = min(self.index_topk, seqlen)
        # [batch, seqlen, index_topk]
        topk_indices = index_scores.topk(topk_k, dim=-1)[1]

        return index_scores, topk_indices


# fused-lightning-indexer
def fused_lightning_indexer(q: torch.Tensor,
                            k: torch.Tensor,
                            weights: torch.Tensor,
                            index_topk,
                            actual_seq_qlen=None,
                            actual_seq_klen=None,
                            layout_query='BSND',
                            layout_key='BSND',
                            ):
    q = rearrange(q, 's b h d -> b s h d').to(torch.bfloat16)
    k = rearrange(k, 's b h d -> b s h d').to(torch.bfloat16)
    weights = rearrange(weights, 's b d -> b s d').to(torch.bfloat16)

    topk_indices, topk_score = torch_npu.npu_lightning_indexer(
        q,
        k,
        weights,
        actual_seq_lengths_query=actual_seq_qlen,
        actual_seq_lengths_key=actual_seq_klen,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=index_topk,
        sparse_mode=3,
        return_value=True,
    )
    topk_indices = topk_indices.squeeze(2)
    topk_score = topk_score.squeeze(2)
    return topk_indices, topk_score


def fused_npu_sparse_flash_attention(query, key, value, topk_indices, query_rope, key_rope, softmax_scale):
    # ===================================
    # Run sparse attention kernel
    # ===================================
    # use fused sparse_flash_attention
    query, key, value = [
        rearrange(x, 's b n d -> b s n d')
        for x in [query, key, value]
    ]

    topk_indices = topk_indices.unsqueeze(2)

    query_rope = rearrange(query_rope, 's b h d -> b s h d')
    key_rope = rearrange(key_rope, 's b h d -> b s h d')

    actual_seq_len = torch.tensor([query.shape[1]], dtype=torch.int32, device=query.device)

    output, softmax_max, softmax_sum, *_ = torch_npu.npu_sparse_flash_attention(
        query, key, value,
        sparse_indices=topk_indices.to(torch.int32),
        block_table=None,
        actual_seq_lengths_query=actual_seq_len,
        actual_seq_lengths_kv=actual_seq_len,
        query_rope=query_rope,
        key_rope=key_rope,
        scale_value=softmax_scale,
        sparse_block_size=1,
        layout_query='BSND',
        layout_kv='BSND',
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=True,
    )

    output = rearrange(output, 'b s h d -> s b h d')

    return output, softmax_max, softmax_sum


#fused-dsa-indexer-kl_loss
def fused_compute_dsa_indexer_kl_loss(
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    q_pos_emb,
    k_pos_emb,
    query_index,
    key_index,
    weights,
    softmax_max,
    softmax_sum,
    packed_seq_params,
    tensor_model_parallel_size=1,
) -> torch.Tensor:
    """
    Compute KL divergence loss between index_scores and true attention_scores.

    This loss trains the indexer to predict which tokens are important by matching the distribution
    of true attention scores.

    Reference: Section 2.1 of
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

    Args:
        index_scores: Scores predicted by indexer [batch, seqlen_q, seqlen_k].
        topk_indices: Top-k indices [batch, seqlen_q, index_topk].
        query: Query tensor [seqlen_q, batch, heads, dim].
        key: Key tensor [seqlen_k, batch, heads, dim].
        softmax_scale: Scale coefficient after q @ k^T.
        loss_coeff: Coefficient for the indexer KL divergence loss.
        sparse_loss: bool, whether to use sparse indexer loss. If True, only the topk
            indices will be used to compute the loss.
        pg_collection: Process group collection, must have TP process group.

    Returns:
        index_loss: KL divergence loss (scalar).
    """

    args = get_args()

    if tensor_model_parallel_size > 1:
        tp_group = mpu.get_tensor_model_parallel_group()
        total_query = allgather_head_dim(query, tensor_model_parallel_size, tp_group)
        total_query_rope = allgather_head_dim(q_pos_emb, tensor_model_parallel_size, tp_group)

        softmax_max = gather_from_tensor_model_parallel_region(softmax_max)
        softmax_sum = gather_from_tensor_model_parallel_region(softmax_sum)
    else:
        total_query = query
        total_query_rope = q_pos_emb

    if args.context_parallel_size > 1 and args.context_parallel_algo == 'kvallgather_cp_algo':
        cp_group = mpu.get_context_parallel_group()
        cp_stream = torch.npu.Stream(device=torch.npu.current_device())

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
            actual_seq_qlen=None,
            actual_seq_klen=None,
            layout='BSND',
            cp_group=cp_group,
            cp_stream=cp_stream,
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
            actual_seq_qlen=None,
            actual_seq_klen=None,
            layout='BSND',
        )

    indexer_loss = loss * loss_coeff
    return indexer_loss


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
        *,
        query_rope=None,
        key_rope=None,
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout='BSND',
        sparse_mode=3,
        pre_tokens=65536,
        next_tokens=65536,
):
    """NPU Sparse Lightning Indexer KL Divergence Loss Function"""

    query, key, query_index, key_index, weights = [
        x.transpose(0, 1)
        for x in [query, key, query_index, key_index, weights]
    ]

    topk_indices = topk_indices.unsqueeze(2)
    if query_rope is not None:
        query_rope, key_rope = [x.transpose(0, 1) for x in [query_rope, key_rope]]

    sq = query.shape[1]
    loss = LILossTrain.apply(query, key, query_index, key_index, weights, topk_indices, softmax_max, softmax_sum,
                             scale_value, query_rope, key_rope, actual_seq_qlen, actual_seq_klen, layout, sparse_mode,
                             pre_tokens, next_tokens, )
    return loss / sq


class LILossTrain(torch.autograd.Function):
    """
    A custom autograd function that computes kl loss in sparse lightning indexer.

    This interface implements the backward functionality of npu_lightning_indexer and integrates the loss computation.
    The npu_lightning_indexer selects the top-k pairs between queries and keys in attention that exhibit the strongest
    intrinsic correlations, storing them in sparse_indices. This reduces the computational cost of attention in
    long-sequence scenarios and improves training performance.
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
        """
        Forward pass: compute the total loss by processing hidden states in chunks.

        Args:
            ctx: Context object used to save tensors for backward pass.
            query (Tensor): Required. Represents the Attention query. Shapes: (B, S1, N1, D), (T1, N1, D)
            key (Tensor): Required. Represents the Attention key. Shapes: (B, S2, N2, D), (T2, N2, D)
            query_index (Tensor): Required. Input query for the lightning_indexer forward pass.
            key_index (Tensor): Required. Input key for the lightning_indexer forward pass.
            weights (Tensor): Required. Weight coefficients of lightning_indexer.
            sparse_indices (Tensor): Required. Token indices of sorted key and key_index.
            softmax_max (Tensor): Required. Maximum values from Attention softmax results.
            softmax_sum (Tensor): Required. Sum values from Attention softmax results.
            scale_value (float): Required scaling coefficient.
            query_rope (Tensor, optional): RoPE information for query in MLA architecture.
            key_rope (Tensor, optional): RoPE information for key in MLA architecture.
            actual_seq_qlen (list[int], optional): Required in TND layout. Cumulative sequence lengths for query.
            actual_seq_klen (list[int], optional): Required in TND layout. Cumulative sequence lengths for key.
            layout (str, optional): Input data layout format. Supported: "BSND", "TND". Default: "BSND".
            sparse_mode (int, optional): Sparse computation mode. Default: 3.
            pre_tokens (int, optional): Number of preceding tokens for sparse Attention. Default: 65536.
            next_tokens (int, optional): Number of succeeding tokens for sparse Attention. Default: 65536.
        Returns:
            d_query_index (Tensor): Gradient of query_index.
            d_key_index (Tensor): Gradient of key_index.
            d_weights (Tensor): Gradient of weights.
            loss (Tensor): Difference between network forward output and golden value.
        """

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

        # Save computed gradients for use in backward pass
        ctx.save_for_backward(d_query_index, d_key_index, d_weights)
        return loss[0]

    @staticmethod
    def backward(ctx, *grad_output) -> Tuple:
        """
        Backward pass: propagate upstream gradients through the precomputed gradients.

        Args:
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient output.

        Returns:
            tuple: Gradients.
        """
        d_query_index, d_key_index, d_weights = ctx.saved_tensors
        grad_scale = grad_output[0]
        if torch.ne(grad_scale, torch.tensor(1.0, device=grad_scale.device)):
            d_query_index = d_query_index * grad_scale
            d_key_index = d_key_index * grad_scale
            d_weights = d_weights * grad_scale

        res_list = [None] * 12
        return None, None, d_query_index, d_key_index, d_weights, *res_list

