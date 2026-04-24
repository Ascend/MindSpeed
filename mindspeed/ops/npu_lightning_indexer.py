import torch

from mindspeed.op_builder.npu_lightning_indexer_builder import NPULightningIndexerOpBuilder

__all__ = ["npu_lightning_indexer"]

op_builder = NPULightningIndexerOpBuilder()

class LightningIndexer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, weights, cu_seq_lens_q, cu_seq_lens_k, layout, 
                sparse_count, sparse_mode, pre_tokens, next_tokens, cmp_ratio):
        op = op_builder.load()

        sparse_indices, sparse_values = op.npu_lightning_indexer(
            query,
            key,
            weights,
            cu_seq_lens_q,
            cu_seq_lens_k,
            None,  # BlockTable for inference
            layout,
            layout,
            sparse_count,
            sparse_mode,
            pre_tokens,
            next_tokens,
            cmp_ratio,
            True,  # returnValues
        )

        ctx.save_for_backward(query, key, weights, cu_seq_lens_q, cu_seq_lens_k, sparse_indices)
        ctx.layout = layout
        ctx.sparse_mode = sparse_mode
        ctx.pre_tokens = pre_tokens
        ctx.next_tokens = next_tokens
        ctx.cmp_ratio = cmp_ratio

        return sparse_indices, sparse_values

    @staticmethod
    def backward(ctx, _, grad_output):
        op = op_builder.load()
        query, key, weights, cu_seq_lens_q, cu_seq_lens_k, sparse_indices = ctx.saved_tensors
        query_grad, k_grad, weights_grad = op.npu_lightning_indexer_grad(
            query,
            key,
            grad_output,
            sparse_indices,
            weights,
            cu_seq_lens_q,
            cu_seq_lens_k,
            ctx.layout,
            ctx.sparse_mode,
            ctx.pre_tokens,
            ctx.next_tokens,
            ctx.cmp_ratio
        )
        return query_grad, k_grad, weights_grad, None, None, None, None, None, None, None, None


def npu_lightning_indexer(query, key, weights, 
                          layout="BSND", cu_seq_lens_q=None, cu_seq_lens_k=None,
                          sparse_count=2048, sparse_mode=3, 
                          pre_tokens=2**63-1, next_tokens=2**63-1, cmp_ratio=1):
    cu_seq_lens_q = cu_seq_lens_k = None  # not support TND
    return LightningIndexer.apply(query, key, weights, cu_seq_lens_q, cu_seq_lens_k, layout,
                                  sparse_count, sparse_mode, pre_tokens, next_tokens, cmp_ratio)

