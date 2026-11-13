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
import torch_npu

from mindspeed.op_builder import MoeTokenUnpermuteOpBuilder

__all__ = ["npu_moe_token_unpermute"]


moe_token_unpermute_op_builder = MoeTokenUnpermuteOpBuilder()


def npu_moe_token_unpermute(
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        probs: torch.Tensor = None,
        padded_mode: bool = False,
        restore_shape: torch.Size = None,
):
    moe_token_unpermute_ops = moe_token_unpermute_op_builder.load()
    return moe_token_unpermute_ops.npu_moe_token_unpermute(
        permuted_tokens, sorted_indices, probs, padded_mode, restore_shape)
