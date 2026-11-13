# coding=utf-8
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


class Config:
    def __init__(self,
                 hidden_size,
                 num_experts=1,
                 ep_size=1,
                 topk=1,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 aux_loss_coef=0.0,
                 z_loss_coef=0.0,
                 noisy_gate_policy=None,
                 no_drop=False,
                 dynamic_padding=False,
                 use_sinkhorn=False,
                 sequence_parallel=False,
                 reshape_index_select=None,
                 ):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.topk = topk
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef
        self.noisy_gate_policy = noisy_gate_policy
        self.no_drop = no_drop
        self.dynamic_padding = dynamic_padding
        self.use_sinkhorn = use_sinkhorn
        self.dynamic_capacity = None
        self.sequence_parallel = sequence_parallel
        self.reshape_index_select = None
        if reshape_index_select:
            self.reshape_index_select = torch.tensor(reshape_index_select, dtype=torch.int32,
                                                     device=torch.npu.current_device())
