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
from mindspeed.op_builder import FusedEmaAdamWOpBuilder

__all__ = ["npu_apply_fused_ema_adamw"]

fused_ema_adamw_op_builder = FusedEmaAdamWOpBuilder()


def npu_apply_fused_ema_adamw(grad: torch.Tensor,
                              var: torch.Tensor,
                              m: torch.Tensor,
                              v: torch.Tensor,
                              s: torch.Tensor,
                              step: torch.Tensor,
                              lr: float = 1e-3,
                              ema_decay: float = 0.9999,
                              beta1: float = 0.9,
                              beta2: float = 0.999,
                              eps: float = 1e-8,
                              mode: int = 1,
                              bias_correction: bool = True,
                              weight_decay: float = 0.0):
    fused_ema_adamw_ops = fused_ema_adamw_op_builder.load()
    return fused_ema_adamw_ops.npu_apply_fused_ema_adamw(grad,
                                                         var,
                                                         m,
                                                         v,
                                                         s,
                                                         step,
                                                         lr,
                                                         ema_decay,
                                                         beta1,
                                                         beta2,
                                                         eps,
                                                         mode,
                                                         bias_correction,
                                                         weight_decay)