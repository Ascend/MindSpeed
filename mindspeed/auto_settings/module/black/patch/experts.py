# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
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
