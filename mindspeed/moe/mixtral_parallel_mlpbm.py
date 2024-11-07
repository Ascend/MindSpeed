# coding=utf-8
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
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
import torch.nn.functional as F

from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.training import get_args
from megatron.core import parallel_state
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.model.transformer import should_recompute_activation


class MixtralParallelMLPBM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_checkpoint_manager = None
        self.ffn_dim = config.ffn_hidden_size
        self.hidden_dim = config.hidden_size
        self.layer_number = None

        self.w1 = ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
            is_expert=False,
            pipe_experts=get_args().use_pipe_experts
        )

        self.w2 = RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            skip_bias_add=True,
            input_is_parallel=True,
            is_expert=False,
            pipe_experts=get_args().use_pipe_experts
        )

        self.w3 = ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
            is_expert=False,
            pipe_experts=get_args().use_pipe_experts
        )

        self.act_fn = F.silu
        if get_args().use_nanopipe and parallel_state.get_pipeline_model_parallel_world_size() > 1 \
                    and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            setattr(self.w1, "in_nano", True)
            setattr(self.w2, "in_nano", True)
            setattr(self.w3, "in_nano", True)

    def forward(self, hidden_states):
        is_recompute_activation = should_recompute_activation(self.layer_number)

        if is_recompute_activation:
            self.activation_checkpoint_manager = CheckpointWithoutOutput()
            act_intermediate_parallel = self.activation_checkpoint_manager.checkpoint(self.act_fn, False, self.w1(hidden_states)[0])
            current_hidden_states = act_intermediate_parallel * self.w3(hidden_states)[0]
            self.activation_checkpoint_manager.discard_output()
            current_hidden_states = self.w2(current_hidden_states)[0]
            if current_hidden_states.requires_grad:
                current_hidden_states.register_hook(self.activation_checkpoint_manager.recompute)
        else:
            current_hidden_states = self.act_fn(self.w1(hidden_states)[0]) * self.w3(hidden_states)[0]
            current_hidden_states = self.w2(current_hidden_states)[0]

        return current_hidden_states
