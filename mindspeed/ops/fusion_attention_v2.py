# Copyright 2024 Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch_npu

__all__ = ["npu_fusion_attention"]


def npu_fusion_attention(
    query,
    key,
    value,
    head_num,
    input_layout,
    *,
    pse=None,
    padding_mask=None,
    atten_mask=None,
    scale=1.0,
    keep_prob=1.0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    inner_precise=0,
    prefix=None,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    sparse_mode=0,
    gen_mask_parallel=True,
    sync=False,
    pse_type=1,
    q_start_idx=None,
    kv_start_idx=None,
):
    if pse is not None and pse.requires_grad and torch.is_grad_enabled():
        raise ValueError("pse gradients are not supported by npu_fusion_attention_v2")
    return torch_npu.npu_fusion_attention_v2(
        query,
        key,
        value,
        head_num,
        input_layout,
        pse=pse,
        padding_mask=padding_mask,
        atten_mask=atten_mask,
        scale=scale,
        keep_prob=keep_prob,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        inner_precise=inner_precise,
        prefix=prefix,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        sparse_mode=sparse_mode,
        gen_mask_parallel=gen_mask_parallel,
        sync=sync,
        pse_type=pse_type,
        q_start_idx=q_start_idx,
        kv_start_idx=kv_start_idx,
    )


def npu_fusion_attention_grad(
    query,
    key,
    value,
    grad_outputs,
    head_num,
    input_layout,
    *,
    pse=None,
    padding_mask=None,
    atten_mask=None,
    softmax_max=None,
    softmax_sum=None,
    softmax_in=None,
    attention_in=None,
    scale=1.0,
    keep_prob=1.0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    inner_precise=0,
    seed=1234,
    offset=0,
    numels=0,
    prefix=None,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    sparse_mode=0,
    gen_mask_parallel=True,
    sync=False,
    pse_type=1,
    q_start_idx=None,
    kv_start_idx=None,
):
    return torch_npu.npu_fusion_attention_grad_v2(
        query,
        key,
        value,
        grad_outputs,
        head_num,
        input_layout,
        pse=pse,
        padding_mask=padding_mask,
        atten_mask=atten_mask,
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        softmax_in=softmax_in,
        attention_in=attention_in,
        scale_value=scale,
        keep_prob=keep_prob,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        inner_precise=inner_precise,
        seed=seed,
        offset=offset,
        numels=numels,
        prefix=prefix,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        sparse_mode=sparse_mode,
        gen_mask_parallel=gen_mask_parallel,
        sync=sync,
        pse_type=pse_type,
        q_start_idx=q_start_idx,
        kv_start_idx=kv_start_idx,
    )[:4]
