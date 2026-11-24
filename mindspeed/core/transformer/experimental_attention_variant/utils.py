# Copyright 2026 Huawei Technologies Co., Ltd.
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
from einops import rearrange


def get_dsa_query_token_count(query: torch.Tensor, layout: str) -> int:
    """Return the number of query tokens used by the fused DSA loss mean."""
    if layout == 'TND':
        return query.size(0)
    if layout == 'BSND':
        # MindSpeed enters the wrapper in [sequence, batch, heads, dim] order.
        return query.size(0) * query.size(1)
    raise ValueError(f"Unsupported DSA layout: {layout}")


def expand_dsa_kv_heads(tensor: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    """Expand grouped KV heads so a batched QK/AV matmul has one KV head per Q head."""
    num_kv_heads = tensor.size(2)
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "The number of DSA query heads must be divisible by the number of KV heads, "
            f"but got query_heads={num_query_heads}, kv_heads={num_kv_heads}."
        )
    if num_query_heads == num_kv_heads:
        return tensor
    return tensor.repeat_interleave(num_query_heads // num_kv_heads, dim=2)


def allgather_head_dim(data_ag, tp, tp_group, layout='BSND'):
    """Allgather along head dimension for TP communication."""
    if layout == 'TND':
        data_ag = rearrange(data_ag, 't h d -> h t d')
    else:
        data_ag = rearrange(data_ag, 's b h d -> h s b d')
    data_ag_shape = list(data_ag.size())
    data_ag_shape[0] *= tp
    total_data_ag = torch.empty(data_ag_shape, dtype=data_ag.dtype, device=data_ag.device)
    torch.distributed.all_gather_into_tensor(total_data_ag, data_ag.contiguous(), group=tp_group)
    if layout == 'TND':
        total_data_ag = rearrange(total_data_ag, 'h t d -> t h d')
    else:
        total_data_ag = rearrange(total_data_ag, 'h s b d -> s b h d')
    return total_data_ag
