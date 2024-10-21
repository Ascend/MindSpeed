# Copyright 2024 Huawei Technologies Co., Ltd
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
# ============================================================================
from functools import wraps
from typing import List

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors
from torch._utils import _unflatten_dense_tensors

from megatron.core.transformer import TransformerConfig
from megatron.core.utils import get_attr_wrapped_model
from megatron.training import get_args
from mindspeed.core.tensor_parallel.comm_group_api import TPXCollectiveComm
from mindspeed.core.tensor_parallel.comm_group_api import TPYCollectiveComm
from mindspeed.core.tensor_parallel.tp_2d.layernorm_2d import LayerNorm2D
from mindspeed.core.tensor_parallel.tp_2d.rms_norm_2d import RMSNorm2D


def _allreduce_layernorm_grads_wrapper(function):
    @wraps(function)
    def wrapper(model: List[torch.nn.Module], config: TransformerConfig):
        function(model, config)
        layer_norm_2d_grads = []
        for model_chunk in model:
            for name, param in get_attr_wrapped_model(model_chunk, "named_parameters")():
                if param.requires_grad and getattr(param, "2d_tp", False):
                    layer_norm_2d_grad = param.main_grad
                    layer_norm_2d_grads.append(layer_norm_2d_grad.data)

        if layer_norm_2d_grads:
            coalesced = _flatten_dense_tensors(layer_norm_2d_grads)
            torch.distributed.all_reduce(coalesced, group=TPXCollectiveComm.get_comm_group())
            for buf, synced in zip(
                layer_norm_2d_grads, _unflatten_dense_tensors(coalesced, layer_norm_2d_grads)
            ):
                buf.copy_(synced)

    return wrapper


def get_norm_tp_2d(config):
    args = get_args()
    if args.normalization == "LayerNorm":
        return LayerNorm2D(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            last_dim_split_comm_intf=TPYCollectiveComm(),
        )
    elif args.normalization == "RMSNorm":
        if args.apply_layernorm_1p:
            raise NotImplementedError(
                "RMSNorm does not currently support the layernorm_1p formulation."
            )
        return RMSNorm2D(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            last_dim_split_comm_intf=TPYCollectiveComm(),
        )
    else:
        raise Exception(f"unsupported norm type '{args.normalization}'.")
