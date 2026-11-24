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


__all__ = ["npu_moe_token_unpermute"]


def npu_moe_token_unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
    padded_mode: bool = False,
    restore_shape: torch.Size = None,
):
    if (
        not padded_mode
        and probs is not None
        and probs.requires_grad
        and torch.is_grad_enabled()
        and permuted_tokens.shape[0] < sorted_indices.numel()
    ):
        # The inverse permutation identifies slots removed by token dropping.
        # WhereBackward clears their gradients, including uninitialized NaNs.
        retained = sorted_indices.reshape_as(probs) < permuted_tokens.shape[0]
        probs = torch.where(retained, probs, 0)
    return torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs, padded_mode, restore_shape)
