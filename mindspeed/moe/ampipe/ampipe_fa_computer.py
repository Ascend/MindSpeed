# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from megatron.training import get_args
from mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel_for_ampipe import (attn_with_cp_for_ampipe_forward,
                                                                                                    attn_with_cp_for_ampipe_backward)
from mindspeed.moe.ampipe.ampipe_args import FlashAttentionSaveForBwdArgs, FACpFwdArgs
from mindspeed.moe.ampipe.ampipe_fa import flash_attn_forward, flash_attn_backward


class FlashAttentionComputer:
    def __init__(self, fa_fwd_args, fa_bwd_args=None):
        super().__init__()
        self.fa_bwd_args = fa_bwd_args
        self.fa_fwd_args = fa_fwd_args
        self.context_parallel = get_args().context_parallel_size > 1

    def forward(self, ctx, q, k, v, attention_mask):
        global_args = get_args()
        flash_tensor_list = self.fa_fwd_args.flash_tensor_list
        cur_degree = self.fa_fwd_args.cur_degree

        if self.context_parallel:
            if cur_degree == 0:
                flash_args_save_for_bwd = FlashAttentionSaveForBwdArgs()
                ctx.flash_args.append(flash_args_save_for_bwd)
            fa_cp_fwd_args = FACpFwdArgs(q, k, v)
            cur_attn_out = attn_with_cp_for_ampipe_forward(ctx.flash_args[0],
                                                           fa_cp_fwd_args=fa_cp_fwd_args,
                                                           fa_fwd_args=self.fa_fwd_args)
        else:
            flash_args_save_for_bwd = FlashAttentionSaveForBwdArgs()
            q_token_start_idx = self.fa_fwd_args.q_token_start_idx
            q_token_end_idx = q_token_start_idx + self.fa_fwd_args.chunk_len
            next_tokens = q_token_start_idx
            q_use = q[q_token_start_idx:q_token_end_idx]
            cur_attn_mask = attention_mask[q_token_start_idx:q_token_end_idx]
            output_chunk = flash_attn_forward((q_use, k, v, self.fa_fwd_args.head_num),
                                              attn_mask=cur_attn_mask,
                                              softmax_scale=self.fa_fwd_args.softmax_scale,
                                              sparse_mode=self.fa_fwd_args.sparse_mode,
                                              next_tokens=next_tokens)
            cur_attn_out, cur_softmax_max, cur_softmax_sum = output_chunk[0], output_chunk[1], output_chunk[2]
            flash_tensor_list.extend([q_use, cur_attn_mask, cur_softmax_max, cur_softmax_sum])
            flash_args_save_for_bwd.next_tokens = next_tokens
            ctx.flash_args.append(flash_args_save_for_bwd)
        # 内存优化
        self._optimize_attn_memory(k, v)
        # 提前做一次mlp的allgather
        should_do_allgather_in_attention = (
                cur_degree == global_args.ampipe_degree - 1
                and global_args.sequence_parallel
                and global_args.ampipe_tp_sp_comm_overlap
                and not global_args.pipe_experts_multi_stream
        )
        if should_do_allgather_in_attention:
            ctx.async_comm.fw_all_gather_not_multistream()
        # attention后的matmul (RowParallelLinear)
        detach_attn_out = cur_attn_out.detach()
        detach_attn_out.requires_grad = True
        with torch.enable_grad():
            attn_dense_out, attn_bias = self.fa_fwd_args.attention_dense(detach_attn_out)
        return detach_attn_out, attn_dense_out, attn_bias

    def backward(self, ctx, grad_output):
        # attention dense 反向
        c = self.fa_bwd_args.cur_degree
        dense_list_slice_len = self.fa_bwd_args.dense_tensor_list_len // ctx.pipe_degree
        cur_attn_out, attn_dense_out = self.fa_bwd_args.dense_tensor_list[
                                       c * dense_list_slice_len:(c + 1) * dense_list_slice_len
                                       ]
        if self.context_parallel and c == ctx.pipe_degree - 1:
            next_attn_out = self.fa_bwd_args.dense_tensor_list[0]
            attn_out_all = torch.cat((next_attn_out.unsqueeze(0), cur_attn_out.unsqueeze(0)), dim=0)
            self.fa_bwd_args.attn_out_all = attn_out_all
        attn_dense_out.backward(grad_output)
        grad_flash = cur_attn_out.grad
        del self.fa_bwd_args.dense_tensor_list[c * dense_list_slice_len:(c + 1) * dense_list_slice_len]

        # FA反向
        flash_tensor_list = self.fa_bwd_args.flash_tensor_list
        if self.context_parallel:
            self.fa_bwd_args.cur_degree = ctx.pipe_degree - 1 - c
            grad_attention = attn_with_cp_for_ampipe_backward(
                ctx.flash_args[0], self.fa_bwd_args.attn_out_all, flash_tensor_list, grad_flash,
                self.fa_bwd_args
            )
            grad_q, grad_k, grad_v = grad_attention[0], grad_attention[1], grad_attention[2]
        else:
            grad_q, grad_k, grad_v = self.fa_bwd_args.grad_q, self.fa_bwd_args.grad_k, self.fa_bwd_args.grad_v
            fa_list_slice_len = (self.fa_bwd_args.flash_tensor_list_len - 2) // ctx.pipe_degree
            q, cur_attn_mask, cur_softmax_max, cur_softmax_sum = flash_tensor_list[
                                                                 c * fa_list_slice_len:(c + 1) * fa_list_slice_len
                                                                 ]
            softmax_scale = self.fa_fwd_args.softmax_scale
            grad_attention = flash_attn_backward(
                (q, self.fa_bwd_args.k, self.fa_bwd_args.v, ctx.head), grad_flash,
                cur_softmax_max, cur_softmax_sum, cur_attn_out, cur_attn_mask, softmax_scale,
                next_tokens=ctx.flash_args[c].next_tokens
            )
            d_q, d_k, d_v = grad_attention[0], grad_attention[1], grad_attention[2]
            grad_k = grad_k + d_k if grad_k is not None else d_k
            grad_v = grad_v + d_v if grad_v is not None else d_v
            grad_q.insert(0, d_q)
            self.fa_bwd_args.grad_q, self.fa_bwd_args.grad_k, self.fa_bwd_args.grad_v = grad_q, grad_k, grad_v
        return grad_q, grad_k, grad_v

    def _optimize_attn_memory(self, k, v):
        if self.fa_fwd_args.cur_degree == get_args().ampipe_degree - 1:
            if self.context_parallel:
                for i, kv in enumerate(self.fa_fwd_args.kv_list):
                    if i < len(self.fa_fwd_args.kv_list) - 1:
                        kv.untyped_storage().resize_(0)
                k.untyped_storage().resize_(0)
                v.untyped_storage().resize_(0)
                self.fa_fwd_args.kv_list.clear()
                self.fa_fwd_args.o_max_sum_list.clear()
            else:
                self.fa_fwd_args.flash_tensor_list.append(k)
                self.fa_fwd_args.flash_tensor_list.append(v)
