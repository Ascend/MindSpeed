# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
import torch.distributed as dist
from einops import rearrange
from megatron.training import get_args

from mindspeed.ops.npu_ring_attention_update import npu_ring_attention_update


class RingP2P:
    def __init__(self, ring_global_ranks, group, group_for_send_recv_overlap=None, is_backward=False) -> None:
        self.group = group
        self.group_for_send_recv_overlap = group
        if group_for_send_recv_overlap is not None:
            self.group_for_send_recv_overlap = group_for_send_recv_overlap

        global_rank = dist.get_rank()
        ring_rank = ring_global_ranks.index(global_rank)
        ring_size = len(ring_global_ranks)
        self.next = ring_global_ranks[(ring_rank + 1) % ring_size]
        self.prev = ring_global_ranks[(ring_rank + ring_size - 1) % ring_size]
        self.ring_rank = ring_rank
        if is_backward:
            self.next, self.prev = self.prev, self.next

        self.send_recv_ops = []
    
    def async_send_recv(self, send_tensor, recv_tensor):
        if self.ring_rank % 2 == 0:
            send_op = dist.isend(send_tensor, self.next, self.group)
            recv_op = dist.irecv(recv_tensor, self.prev, self.group_for_send_recv_overlap)
            self.send_recv_ops.append(send_op)
            self.send_recv_ops.append(recv_op)
        else:
            recv_op = dist.irecv(recv_tensor, self.prev, self.group)
            send_op = dist.isend(send_tensor, self.next, self.group_for_send_recv_overlap)
            self.send_recv_ops.append(recv_op)
            self.send_recv_ops.append(send_op)
    
    def wait(self):
        if len(self.send_recv_ops) > 0:
            for op in self.send_recv_ops:
                op.wait()
            self.send_recv_ops = []
            return 1
        else:
            return 0


def forward_update_without_fused(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                                 cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen=None, layout='SBH'):
    # update softmax_max
    origin_dtype = prev_attn_out.dtype
    softmax_max = torch.maximum(prev_softmax_max, cur_softmax_max)
    prev_scale = torch.exp(prev_softmax_max - softmax_max)
    cur_scale = torch.exp(cur_softmax_max - softmax_max)

    # update softmax_sum
    prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
    cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
    softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled

    # out updating scale
    prev_out_scale = prev_softmax_sum_scaled / softmax_sum
    cur_out_scale = cur_softmax_sum_scaled / softmax_sum

    # [b, n, s, 8] -> [s, b, h]
    # SBH layout
    n = prev_out_scale.shape[1]
    h = prev_attn_out.shape[-1]
    d = h // n
    prev_out_scale = prev_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d)
    prev_out_scale = rearrange(prev_out_scale, 'b n s d -> s b (n d)').contiguous()
    cur_out_scale = cur_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d)
    cur_out_scale = rearrange(cur_out_scale, 'b n s d -> s b (n d)').contiguous()

    # update output
    attn_out = prev_attn_out * prev_out_scale + cur_attn_out * cur_out_scale
    attn_out = attn_out.to(origin_dtype)
    return attn_out, softmax_max, softmax_sum


def forward_update(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                   cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen=None, layout='SBH'):
    """
    Updates the attention output and softmax statistics for the ring attention mechanism,
    with added parameters for enhanced flexibility and extensibility.

    This function is designed to update the attention output and related softmax statistics
    for a given sequence length in a ring attention mechanism. It handles the merging of
    previous and current attention outputs and their corresponding softmax statistics.
    The introduction of `actual_seq_qlen` and `layout` parameters allows for greater flexibility
    in handling variable sequence lengths and different tensor layouts, respectively.

    Parameters:
    - prev_attn_out (Tensor): The attention output from the previous process.
    - prev_softmax_max (Tensor): The maximum value of the softmax distribution from the previous process.
    - prev_softmax_sum (Tensor): The sum of the softmax distribution from the previous process.
    - cur_attn_out (Tensor): The attention output from the current process.
    - cur_softmax_max (Tensor): The maximum value of the softmax distribution from the current process.
    - cur_softmax_sum (Tensor): The sum of the softmax distribution from the current process.
    - actual_seq_qlen (Tensor, optional): The actual sequence length for the query. This parameter
                                      is crucial for handling variable-length sequences and ensuring
                                      that the attention mechanism operates correctly under such conditions.
                                      If not provided, it defaults to the length of the current attention output.
    - layout (str, optional): The layout format of the input tensors. This parameter allows for the specification
                              of different tensor layouts, enhancing the function's versatility across various
                              model architectures. Default is 'SBH', where:
        - S: Sequence length
        - B: Batch size
        - H: Hidden size (number of attention heads)

    Returns:
    - updated_attn_out (Tensor): The updated attention output after merging previous and current process.
    - updated_softmax_max (Tensor): The updated maximum value of the softmax distribution.
    - updated_softmax_sum (Tensor): The updated sum of the softmax distribution.
    """
    _args = get_args()
    if hasattr(_args, 'use_fused_ring_attention_update') and _args.use_fused_ring_attention_update:
        return npu_ring_attention_update(prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out,
                                         cur_softmax_max, cur_softmax_sum, actual_seq_qlen, layout)
    return forward_update_without_fused(prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out,
                                    cur_softmax_max, cur_softmax_sum, actual_seq_qlen, layout)


def causal_out_update(q_block_id, kv_block_id, cur_attn_outs, global_attn_outs,
                                   q_index=None, cur_sub_out_seq_len=None):
    cur_attn_out, cur_softmax_max, cur_softmax_sum = cur_attn_outs[0], cur_attn_outs[1], cur_attn_outs[2]
    attn_out, softmax_max, softmax_sum, rng_states = global_attn_outs
    layout = 'SBH'
    # (seed, offset, numels)
    if len(cur_attn_outs) > 3:
        rng_states[kv_block_id] = (cur_attn_outs[4], cur_attn_outs[5], cur_attn_outs[6])

    if q_block_id == kv_block_id:
        attn_out = cur_attn_out
        softmax_max = cur_softmax_max
        softmax_sum = cur_softmax_sum
    elif kv_block_id <= q_block_id:
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            attn_out, softmax_max, softmax_sum,
            cur_attn_out, cur_softmax_max, cur_softmax_sum, layout=layout
        )
        attn_out, softmax_max, softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated
    else:
        # [2s, b, h] -> [2, s, b, h]
        attn_out = attn_out.view(2, attn_out.shape[0] // 2, *attn_out.shape[1:])
        # [b, n, 2s, 8] -> [b, n, 2, s, 8]
        softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1],
                                        2, softmax_max.shape[2] // 2, softmax_max.shape[-1])
        softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1],
                                        2, softmax_sum.shape[2] // 2, softmax_sum.shape[-1])
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            attn_out[1], softmax_max[:, :, 1, :, :], softmax_sum[:, :, 1, :, :],
            cur_attn_out, cur_softmax_max, cur_softmax_sum, layout=layout
        )
        attn_out[1].copy_(attn_out_updated)
        softmax_max[:, :, 1, :, :].copy_(softmax_max_updated)
        softmax_sum[:, :, 1, :, :].copy_(softmax_sum_updated)
        # [2, s, b, h] -> [2s, b, h]
        attn_out = attn_out.view(-1, *attn_out.shape[2:])
        # [b, n, 2, s, 8] -> [b, n, 2s, 8]
        softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1], -1,
                                        softmax_max.shape[-1])
        softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1], -1,
                                        softmax_sum.shape[-1])
    
    return [attn_out, softmax_max, softmax_sum, rng_states]


def general_out_update(q_block_id, kv_block_id, cur_attn_outs, global_attn_outs, cur_sub_out_seq_len=None):
    cur_attn_out, cur_softmax_max, cur_softmax_sum = cur_attn_outs[0], cur_attn_outs[1], cur_attn_outs[2]
    attn_out, softmax_max, softmax_sum, rng_states = global_attn_outs
    layout = 'SBH'
    # (seed, offset, numels)
    rng_states[kv_block_id] = (cur_attn_outs[4], cur_attn_outs[5], cur_attn_outs[6])
    if q_block_id == kv_block_id:
        attn_out = cur_attn_out
        softmax_max = cur_softmax_max
        softmax_sum = cur_softmax_sum
    else:
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            attn_out, softmax_max, softmax_sum,
            cur_attn_out, cur_softmax_max, cur_softmax_sum, layout=layout
        )
        attn_out, softmax_max, softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated
    
    return [attn_out, softmax_max, softmax_sum, rng_states]

        
