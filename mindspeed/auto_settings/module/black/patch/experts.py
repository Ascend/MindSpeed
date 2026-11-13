# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
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
from functools import wraps

import torch
from megatron.training.global_vars import get_args


def sequential_mlp_forward_decorator(fn):

    @wraps(fn)
    def wrapper(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        prof_file = get_args().prof_file
        if prof_file:
            output_local = torch.zeros_like(permuted_local_hidden_states)
            output_bias_local = None
            if self.add_bias:
                output_bias_local = torch.zeros_like(permuted_local_hidden_states)

            cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
            # Insert zero at the begining for offset index's convenience
            zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)
            cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))
            for expert_num, expert in enumerate(self.local_experts):
                start = cumsum_num_tokens[expert_num]
                end = cumsum_num_tokens[expert_num + 1]
                hidden = permuted_local_hidden_states[start:end]

                activation_func_1 = torch.nn.Hardshrink()
                hidden = activation_func_1(hidden)
                output, output_bias = expert(hidden)
                output = activation_func_1(output)

                output_local[start:end] = output
                if self.add_bias:
                    output_bias = output_bias.expand_as(output)
                    output_bias_local[start:end, :] = output_bias

            return output_local, output_bias_local

        return fn(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs)

    return wrapper
