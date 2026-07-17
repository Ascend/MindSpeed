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
from mindspeed.core.tensor_parallel.tp_2d.group_api_2d import TPYCollectiveComm
from mindspeed.core.tensor_parallel.tp_2d.layernorm_2d import LayerNorm2D
from mindspeed.core.tensor_parallel.tp_2d.rms_norm_2d import RMSNorm2D


def get_norm_tp_2d(config, hidden_size=None, eps=None, **kwargs):
    normalization = getattr(config, 'normalization')
    apply_layernorm_1p = getattr(config, 'apply_layernorm_1p', False)
    hidden_size = hidden_size if hidden_size is not None else config.hidden_size
    eps = eps if eps is not None else config.layernorm_epsilon
    if normalization == "LayerNorm":
        return LayerNorm2D(
            hidden_size,
            eps=eps,
            last_dim_split_comm_intf=TPYCollectiveComm(),
        )
    elif normalization == "RMSNorm":
        if apply_layernorm_1p:
            raise NotImplementedError("RMSNorm does not currently support the layernorm_1p formulation.")
        return RMSNorm2D(
            hidden_size,
            eps=eps,
            last_dim_split_comm_intf=TPYCollectiveComm(),
        )
    else:
        raise ValueError(f"unsupported norm type '{normalization}'.")
