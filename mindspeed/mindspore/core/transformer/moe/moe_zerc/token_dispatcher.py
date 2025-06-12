# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.utils import make_viewless_tensor
from megatron.training import get_args


def zerc_preprocess(self, indices: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    ep_size = self.config.expert_model_parallel_size
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_extended_ep_size = ep_size * tp_size

    # ===================================================
    # Calculate input_splits, output_splits for alltoall-v.
    # ===================================================
    self.num_tokens = indices.shape[0]
    total_experts = tp_extended_ep_size * self.num_local_experts
    local_map_info = torch.zeros((self.num_tokens, total_experts), dtype=probs.dtype, device=torch.cuda.current_device(), requires_grad=True)
    local_map_info = torch.scatter(local_map_info, 1, indices, probs)
    local_map_info = local_map_info.view(-1, tp_extended_ep_size, self.num_local_experts) # [num_tokens, tp_extended_ep_size, num_local_experts]
    local_map_info = torch.transpose(local_map_info, 0, 1).contiguous()  # [tp_extended_ep_size, num_tokens, num_local_experts]
    if get_args().moe_tp_extend_ep:
        ep_group = parallel_state.get_tensor_and_expert_parallel_group()
    else:
        ep_group = parallel_state.get_expert_model_parallel_group()
    global_map_info = tensor_parallel.all_to_all(ep_group, local_map_info)
    global_map_info = global_map_info.view(tp_extended_ep_size * self.num_tokens, self.num_local_experts)
    # tk is not a certain number, indicate the number of zero redundancy tokens
    token_send = torch.any(local_map_info, dim=-1) # [ep_size, tk]
    tokens_received = torch.any(global_map_info.view(tp_extended_ep_size, self.num_tokens, self.num_local_experts), dim=-1) # [tp_extended_ep_size, tk]

    self.input_splits = token_send.sum(-1).detach().numpy()  # [tp_extended_ep_size]
    self.output_splits = tokens_received.sum(-1).numpy()  # [tp_extended_ep_size]

    # index for perm1
    self.select_index = torch.nonzero(token_send.ravel()).ravel() % self.num_tokens

    global_map_info = torch.transpose(global_map_info, 1, 0).contiguous() # [num_local_experts, tp_extended_ep_size * num_tokens]
    self.global_map_info = torch.masked_select(global_map_info, torch.any(global_map_info, 0).expand(self.num_local_experts, self.num_tokens * tp_extended_ep_size)).view(self.num_local_experts, -1) # [ep, tk]
    num_tokens_per_local_expert = torch.sum(self.global_map_info.bool(), dim=1)

    self.cuda_sync_point = "no_sync"

    return num_tokens_per_local_expert, self.global_map_info


def zerc_alltoall_token_perm1(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor
):
    input_dtype = hidden_states.dtype
    self.hidden_shape = hidden_states.shape

    assert probs.dim() == 2, "Expected 2D tensor for probs"
    assert indices.dim() == 2, "Expected 2D tensor for indices"
    tokens_per_expert, global_map_info = zerc_preprocess(self, indices, probs)

    # Flatten the input tensor
    # hidden_states: [S/TP, B, H] -> [S*B/TP, H]
    hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

    # Permutation 1: input to AlltoAll input
    self.hiddden_shape_before_permute = hidden_states.shape

    permutated_local_input_tokens = hidden_states.index_select(0, self.select_index)
    permuted_local_probs = None

    return permutated_local_input_tokens.type(input_dtype), permuted_local_probs, tokens_per_expert, global_map_info


def zerc_alltoall_token_perm2(self, global_input_tokens, global_input_token_probs=None, global_map_info=None):
    input_dtype = global_input_tokens.dtype
    self.probs = None
    # Permutation 2: AlltoAll output to expert input if num_local_experts > 1

    self.output_shape = [self.num_tokens, global_input_tokens.shape[1]]
    self.input_shape = global_input_tokens.shape

    self.nr_token_id_recover_probs = torch.nonzero(global_map_info.detach().ravel()).ravel()  # check detach
    self.nr_token_id_recover = self.nr_token_id_recover_probs % (global_map_info.shape[-1])
    global_input_tokens = global_input_tokens.index_select(0, self.nr_token_id_recover)

    if get_args().moe_unperm2_mem_optim:
        global_input_token_probs = torch.index_select(global_map_info.ravel(), 0, self.nr_token_id_recover_probs)

    return global_input_tokens.type(input_dtype), global_input_token_probs


def zerc_alltoall_token_unperm1(
        self,
        hidden_states: torch.Tensor,
        bias: torch.Tensor = None,
        global_map_info: torch.Tensor = None,
):
    assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"

    input_dtype = hidden_states.dtype
    permutated_local_input_tokens = None
    # Unpermutation 2: expert output to AlltoAll input
    if not get_args().moe_unperm2_mem_optim:
        probs = torch.index_select(global_map_info.ravel(), 0, self.nr_token_id_recover_probs)
        hidden_states = hidden_states * probs.unsqueeze(-1)

    permutated_local_input_tokens = torch.zeros(self.input_shape, dtype=hidden_states.dtype, device=torch.cuda.current_device(), requires_grad=True)
    permutated_local_input_tokens = torch.index_add(permutated_local_input_tokens, 0, self.nr_token_id_recover, hidden_states)

    return permutated_local_input_tokens.type(input_dtype)


def zerc_alltoall_token_unperm2(self, permutated_local_input_tokens, probs=None):
    input_dtype = permutated_local_input_tokens.dtype

    output = torch.zeros(self.output_shape, dtype=permutated_local_input_tokens.dtype, device=torch.cuda.current_device())
    output = torch.index_add(output, 0, self.select_index, permutated_local_input_tokens)

    output = output.type(input_dtype)
    # Reshape the output tensor
    output = output.view(self.hidden_shape)

    output = make_viewless_tensor(
        inp=output, requires_grad=output.requires_grad, keep_graph=True
    )
    return output, None
