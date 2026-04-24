import torch

from mindspeed.op_builder.npu_sparse_attn_shared_kv_builder import NPUSparseAttnSharedKVOpBuilder

op_builder = NPUSparseAttnSharedKVOpBuilder()


class SparseAttnSharedKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, ori_kv, cmp_kv, cu_seq_lens_q, cu_seq_lens_ori_kv, cu_seq_lens_cmp_kv, ori_sparse_indices,
                cmp_sparse_indices, sinks, softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left,
                ori_win_right, num_heads_q, num_heads_kv, head_dim, batch_size, max_seq_len_q, max_seq_len_kv, topk,
                layout_q, layout_kv):
        op = op_builder.load()

        metadata = op.npu_sparse_attn_shared_kv_metadata(
            cu_seq_lens_q if cu_seq_lens_q is not None else torch.tensor([]).npu(),
            torch.tensor([]).npu(),  # sequsedOriKv for inference
            torch.tensor([]).npu(),  # sequsedCmpKv for inference
            torch.tensor([]).npu(),  # sequsedQ for inference
            torch.tensor([]).npu(),  # sequsedKv for inference
            num_heads_q,
            num_heads_kv,
            head_dim,
            batch_size,
            max_seq_len_q,
            max_seq_len_kv,
            topk,  # oriTopk not support now
            topk,
            cmp_ratio,
            ori_mask_mode,
            cmp_mask_mode,
            ori_win_left,
            ori_win_right,
            layout_q,
            layout_kv,
            ori_kv is not None,  # hasOriKv
            cmp_kv is not None,  # hasCmpKv
        )

        result, softmax_lse = op.npu_sparse_attn_shared_kv(
            query,
            ori_kv,
            cmp_kv,
            ori_sparse_indices,
            cmp_sparse_indices,
            None,  # oriBlockTable for inference
            None,  # cmpBlockTable for inference
            cu_seq_lens_q,
            cu_seq_lens_ori_kv,
            cu_seq_lens_cmp_kv,
            None,  # sequsedQ for inference
            None,  # sequsedKv for inference
            sinks,
            metadata,
            softmax_scale,
            cmp_ratio,
            ori_mask_mode,
            cmp_mask_mode,
            ori_win_left,
            ori_win_right,
            layout_q,
            layout_kv,
            True,  # returnSoftmaxLse
        )

        ctx.save_for_backward(query, ori_kv, cmp_kv, result, softmax_lse, ori_sparse_indices, cmp_sparse_indices,
                              cu_seq_lens_q, cu_seq_lens_ori_kv, cu_seq_lens_cmp_kv, sinks)
        ctx.softmax_scale = softmax_scale
        ctx.cmp_ratio = cmp_ratio
        ctx.ori_mask_mode = ori_mask_mode
        ctx.cmp_mask_mode = cmp_mask_mode
        ctx.ori_win_left = ori_win_left
        ctx.ori_win_right = ori_win_right
        ctx.layout_q = layout_q
        return result

    @staticmethod
    def backward(ctx, grad_output):
        op = op_builder.load()
        query, ori_kv, cmp_kv, result, softmax_lse, ori_sparse_indices, cmp_sparse_indices, cu_seq_lens_q, \
            cu_seq_lens_ori_kv, cu_seq_lens_cmp_kv, sinks = ctx.saved_tensors
        query_grad, ori_kv_grad, cmp_kv_grad, sinks_grad = op.npu_sparse_attn_shared_kv_grad(
            query,
            ori_kv,
            cmp_kv,
            grad_output,
            result,
            softmax_lse,
            ori_sparse_indices,
            cmp_sparse_indices,
            cu_seq_lens_q,
            cu_seq_lens_ori_kv,
            cu_seq_lens_cmp_kv,
            sinks,
            ctx.softmax_scale,
            ctx.cmp_ratio,
            ctx.ori_mask_mode,
            ctx.cmp_mask_mode,
            ctx.ori_win_left,
            ctx.ori_win_right,
            ctx.layout_q
        )
        return query_grad, ori_kv_grad, cmp_kv_grad, None, None, None, None, None, sinks_grad, None, None, None, None, \
            None, None, None, None, None, None, None, None, None, None, None


def npu_sparse_attn_shared_kv(query, ori_kv, cmp_kv, cmp_sparse_indices, sinks, softmax_scale, cmp_ratio,
                              ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=127, ori_win_right=0):
    cu_seq_lens_q = cu_seq_lens_ori_kv = cu_seq_lens_cmp_kv = None  # not support TND
    ori_sparse_indices = None  # ori kv use band mode
    max_seq_len_q, batch_size, num_heads_q, head_dim = query.size()
    num_heads_kv = 1
    max_seq_len_kv = ori_kv.size(0)
    topk = 0 if cmp_ratio != 4 else cmp_sparse_indices.size(-1)
    layout_q = layout_kv = 'BSND'
    query = query.permute(1, 0, 2, 3).contiguous()  # [S, B, N, D] --> [B, S, N, D]
    ori_kv = ori_kv.permute(1, 0, 2).unsqueeze(2).contiguous()  # [S, B, D] --> [B, S, 1, D]
    cmp_kv = cmp_kv if cmp_kv is None else cmp_kv.permute(1, 0, 2).unsqueeze(2).contiguous()  # [S, B, D] --> [B, S, 1, D]
    cmp_sparse_indices = None if cmp_ratio != 4 else cmp_sparse_indices.unsqueeze(2).contiguous()  # [B, S, K] --> [B, S, 1, K]
    output = SparseAttnSharedKV.apply(query, ori_kv, cmp_kv, cu_seq_lens_q, cu_seq_lens_ori_kv, cu_seq_lens_cmp_kv,
                                      ori_sparse_indices, cmp_sparse_indices, sinks, softmax_scale, cmp_ratio,
                                      ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, num_heads_q,
                                      num_heads_kv, head_dim, batch_size, max_seq_len_q, max_seq_len_kv, topk, layout_q,
                                      layout_kv)
    return output.transpose(0, 1).contiguous()
