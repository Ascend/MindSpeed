import torch

from mindspeed.op_builder.npu_sparse_lightning_indexer_grad_kl_loss_builder import NPUSparseLIGradKlLossOpBuilder

__all__ = [
    "npu_sparse_lightning_indexer_grad_kl_loss",
    ]

op_builder = NPUSparseLIGradKlLossOpBuilder()

class SparseLIGradKlLoss(torch.autograd.Function):
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
            pre_tokens=2147483647,
            next_tokens=2147483647,
            cmp_ratio=1,
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
            layout (str, optional): Input data layout format. Supported: "BSND". Default: "BSND".
            sparse_mode (int, optional): Sparse computation mode. Default: 3.
            pre_tokens (int, optional): Number of preceding tokens for sparse Attention. Default: 65536.
            next_tokens (int, optional): Number of succeeding tokens for sparse Attention. Default: 65536.
            cmp_ratio (int, optional): Compression ratio. Default: 1.
        Returns:
            d_query_index (Tensor): Gradient of query_index.
            d_key_index (Tensor): Gradient of key_index.
            d_weights (Tensor): Gradient of weights.
            loss (Tensor): Difference between network forward output and golden value.
        """
        op = op_builder.load()

        d_query_index, d_key_index, d_weights, loss = op.npu_sparse_lightning_indexer_grad_kl_loss(
            query,
            key,
            query_index,
            key_index,
            weights,
            sparse_indices,
            softmax_max,
            softmax_sum,
            query_rope,
            key_rope,
            actual_seq_qlen,
            actual_seq_klen,
            scale_value,
            layout,
            sparse_mode,
            pre_tokens,
            next_tokens,
            cmp_ratio,
        )

        # Save computed gradients for use in backward pass
        ctx.save_for_backward(d_query_index, d_key_index, d_weights)
        return loss[0]

    @staticmethod
    def backward(ctx, *grad_output):
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

        res_list = [None] * 13
        return None, None, d_query_index, d_key_index, d_weights, *res_list


def npu_sparse_lightning_indexer_grad_kl_loss(
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
        pre_tokens=2147483647,
        next_tokens=2147483647,
        cmp_ratio=1,
):
    """NPU Sparse Lightning Indexer KL Divergence Loss Function"""
    query, key, query_index, key_index, weights = [x.transpose(0, 1) for x in
                                                   [query, key, query_index, key_index, weights]]
    if len(key.shape) == 3:
        key = key.unsqueeze(2)
    topk_indices = topk_indices.unsqueeze(2)
    if query_rope is not None:
        query_rope, key_rope = [x.transpose(0, 1) for x in [query_rope, key_rope]]

    bsz = query.shape[0]
    sq = query.shape[1]
    loss = SparseLIGradKlLoss.apply(
        query, key, query_index, key_index, weights, topk_indices, softmax_max, softmax_sum,
        scale_value, query_rope, key_rope, actual_seq_qlen, actual_seq_klen, layout, sparse_mode,
        pre_tokens, next_tokens, cmp_ratio)
    return loss / (sq * bsz)