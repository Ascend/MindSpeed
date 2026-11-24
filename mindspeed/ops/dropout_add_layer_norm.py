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
import torch_npu
from mindspeed.op_builder import NpuDropoutAddLayerNormOpBuilder

__all__ = ["npu_dropout_add_layer_norm"]


def npu_dropout_add_layer_norm(x0,
                               residual,
                               weight,
                               bias,
                               dropout_p,
                               epsilon,
                               rowscale=None,
                               layerscale=None,
                               prenorm=False,
                               residual_in_fp32=False,
                               return_dropout_mask=False):
    mindspeed_ops = NpuDropoutAddLayerNormOpBuilder().load()
    is_rms_norm = False
    out, pre_norm, dmask = mindspeed_ops.npu_dropout_add_layer_norm(
        x0,
        weight,
        residual,
        bias,
        rowscale,
        layerscale,
        dropout_p,
        epsilon,
        prenorm,
        residual_in_fp32,
        is_rms_norm,
        return_dropout_mask
    )
    if not return_dropout_mask:
        return (
            out.view(x0.shape) if not prenorm else (out.view(x0.shape), pre_norm.view(x0.shape))
        )
    else:
        dmask = (
            dmask.view(x0.shape)
            if dropout_p > 0.0
            else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device)
        )
        return (
            (out.view(x0.shape), dmask)
            if not prenorm
            else (out.view(x0.shape), pre_norm.view(x0.shape), dmask)
        )