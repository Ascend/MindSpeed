# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
from megatron.core import parallel_state, tensor_parallel


def get_zeros_with_tp(input_):
    world_size = parallel_state.get_tensor_model_parallel_world_size()
    zeros_shape = input_.shape[:-1] + (input_.shape[-1] * world_size, )
    return torch.zeros(zeros_shape, dtype=input_.dtype, layout=input_.layout, device=input_.device)


def sequential_mlp_forward(self, permuted_local_hidden_states, tokens_per_expert):
    output_local = get_zeros_with_tp(permuted_local_hidden_states)
    output_bias_local = None
    if self.add_bias:
        output_bias_local = get_zeros_with_tp(permuted_local_hidden_states)

    cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
    # Insert zero at the begining for offset index's convenience
    zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)
    cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))

    if parallel_state.get_tensor_model_parallel_world_size() > 1:
        if not hasattr(self, 'comm_stream'):
            self.comm_stream = torch.cuda.Stream()
        self.comm_stream.wait_stream(torch.cuda.current_stream())

    for expert_num, expert in enumerate(self.local_experts):
        start = cumsum_num_tokens[expert_num]
        end = cumsum_num_tokens[expert_num + 1]
        hidden = permuted_local_hidden_states[start:end]

        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            with torch.cuda.stream(self.comm_stream):
                hidden = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(hidden)
            torch.cuda.current_stream().wait_stream(self.comm_stream)

        output, output_bias = expert(hidden)

        output_local[start:end] = output
        if self.add_bias:
            output_bias = output_bias.expand_as(output)
            output_bias_local[start:end, :] = output_bias

    return output_local, output_bias_local
