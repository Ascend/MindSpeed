# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

from functools import wraps
from torch import Tensor
from megatron.core.packed_seq_params import PackedSeqParams
from mindspeed.utils import get_actual_seq_len, compute_qkv_index, get_position_ids


def gpt_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        actual_seq_len = get_actual_seq_len()

        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=actual_seq_len, 
            cu_seqlens_kv=actual_seq_len
        )

        q_index, kv_index = compute_qkv_index(actual_seq_len.clone().tolist())
        packed_seq_params.q_index = q_index
        packed_seq_params.kv_index = kv_index
        packed_seq_params.position_ids = get_position_ids()

        kwargs['packed_seq_params'] = packed_seq_params
        return fn(*args, **kwargs)

    return wrapper