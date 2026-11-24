# Copyright 2024 Huawei Technologies Co., Ltd.
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

from mindspeed.op_builder import RingAttentionUpdateOpBuilder

__all__ = ["npu_ring_attention_update"]


op_builder = RingAttentionUpdateOpBuilder()


def npu_ring_attention_update(
        prev_attn_out: torch.Tensor,
        prev_softmax_max: torch.Tensor,
        prev_softmax_sum: torch.Tensor,
        cur_attn_out: torch.Tensor,
        cur_softmax_max: torch.Tensor,
        cur_softmax_sum: torch.Tensor,
        actual_seq_qlen: torch.Tensor = None,
        layout: str = "SBH",
):
    ops = op_builder.load()
    return ops.npu_ring_attention_update(
        prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out, cur_softmax_max, cur_softmax_sum,
        actual_seq_qlen, layout)
