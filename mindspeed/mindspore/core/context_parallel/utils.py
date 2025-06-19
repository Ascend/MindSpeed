# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
from mindspeed.core.context_parallel.utils import forward_update


def general_out_update(q_block_id, kv_block_id, cur_attn_outs, global_attn_outs):
    cur_attn_out, cur_softmax_max, cur_softmax_sum = cur_attn_outs[0], cur_attn_outs[1], cur_attn_outs[2]
    attn_out, softmax_max, softmax_sum, rng_states = global_attn_outs
    layout = 'SBH'
    if len(cur_attn_outs) > 3:
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

