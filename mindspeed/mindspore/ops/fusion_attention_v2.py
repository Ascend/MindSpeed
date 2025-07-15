# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
__all__ = ["npu_fusion_attention"]

import torch
from mindspore.ops import auto_generate as gen
from mindspore.ops import speed_fusion_attention as npu_fusion_attention


def npu_fusion_attention_grad(query, key, value, grad_outputs,
                              head_num, input_layout, *, pse=None,
                              padding_mask=None, atten_mask=None,
                              softmax_max=None, softmax_sum=None, softmax_in=None, attention_in=None,
                              scale=1., keep_prob=1., pre_tokens=2147483647,
                              next_tokens=2147483647, inner_precise=0,
                              seed=1234, offset=0, numels=0, prefix=None,
                              actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                              gen_mask_parallel=True, sync=False, pse_type=1, q_start_idx=None,
                              kv_start_idx=None):
    seed = torch.tensor(seed, dtype=torch.int64)
    offset = torch.tensor(offset, dtype=torch.int64)
    numels = torch.tensor(numels, dtype=torch.int64)
    return gen.speed_fusion_attention_grad_op(query, key, value, grad_outputs, head_num, input_layout, pse,
                                              padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in,
                                              attention_in, scale, keep_prob, pre_tokens, next_tokens,
                                              inner_precise, seed, offset, numels, prefix, actual_seq_qlen,
                                              actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync,
                                              pse_type, q_start_idx, kv_start_idx)
