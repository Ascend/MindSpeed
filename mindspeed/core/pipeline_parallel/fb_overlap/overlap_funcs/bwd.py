#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import torch
from megatron.core import parallel_state
from megatron.training import get_args
from megatron.core.transformer.moe.moe_utils import permute
from mindspeed.core.transformer.moe.comm_utils import async_all_to_all, async_all_gather, async_reduce_scatter
from mindspeed.model.transformer import should_recompute_activation
from mindspeed.core.transformer.moe.moe_utils import get_prob_backward_need_tensors
from ..modules.weight_grad_store import WeightGradStore
from ..modules.utils import run_graph_backward


def transformer_layer_backward_moe(
    layer_output_grad,
    layer_graph
):
    self = layer_graph
    args = get_args()
    in_detach_stage = WeightGradStore.is_decoupleBlock
    dispached_input, fc1_out, act_out, probs, indices, global_input_tokens_local_experts_indices = self.recompute_needed_tensors
    ep_group = parallel_state.get_expert_model_parallel_group()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    if args.moe_tp_extend_ep:
        ep_group = parallel_state.get_tensor_and_expert_parallel_group()
    if tp_size > 1:
        shared_expert_grad = layer_output_grad if layer_output_grad is not None else self.unperm2_graph[1].grad
        _, backward_ag_shared, backward_ag_shared_handle = async_all_gather(
            shared_expert_grad, parallel_state.get_tensor_model_parallel_group()
        )
    else:
        backward_ag_shared = layer_output_grad if layer_output_grad is not None else self.unperm2_graph[1].grad
        backward_ag_shared_handle = None

    run_graph_backward(self.unperm2_graph, layer_output_grad, keep_grad=True)
    if backward_ag_shared_handle is not None:
        backward_ag_shared_handle.wait()
        backward_ag_shared_handle = None
        if layer_output_grad is not None:
            layer_output_grad.untyped_storage().resize_(0)
    _, unperm1_out_grad, handle = async_all_to_all(
        self.unperm_a2a_graph[1].grad,
        self.output_splits,
        self.input_splits,
        ep_group
    )
    # overlap alltoall by shared experts backward
    if self.shared_experts_graph[0] is not None:
        run_graph_backward(self.shared_experts_graph, backward_ag_shared)
    if get_args().moe_zero_memory == 'level0' or should_recompute_activation(self.layer.layer_number):
        with torch.no_grad():
            recompute_act_out = self.layer.mlp.experts.activation_func(fc1_out)
            act_out.untyped_storage().resize_(recompute_act_out.untyped_storage().size())
            act_out.untyped_storage().copy_(recompute_act_out.untyped_storage())
            recompute_act_out.untyped_storage().resize_(0)
    handle.wait()
    handle = None

    # recomp permute1 and overlap all2all
    if get_args().moe_zero_memory == 'level0':
        with torch.no_grad():
            input_before_perm1 = self.pre_mlp_layernorm_graph[0]

            def recomp_token_permutation1(hidden_states, indices):
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                permutated_local_input_tokens, _ = permute(
                    hidden_states, indices
                )
                return permutated_local_input_tokens

            perm1_out = recomp_token_permutation1(input_before_perm1, indices)
            _, perm_a2a_out, perm_a2a_handle = async_all_to_all(
                perm1_out,
                self.output_splits,
                self.input_splits,
                ep_group
            )

    run_graph_backward(self.unperm1_graph, unperm1_out_grad)
    WeightGradStore.start_decouple()
    run_graph_backward(self.grouped_mlp_graph, keep_grad=True)  # keep for dw commputation
    if not in_detach_stage:
        WeightGradStore.end_decouple()
    run_graph_backward(self.perm2_graph, keep_graph=True)  # keep for dw commutation
    if get_args().moe_zero_memory == 'level0':
        perm_a2a_handle.wait()
        perm_a2a_handle = None

    _, perm1_out_grad, handle = async_all_to_all(
        self.perm_a2a_graph[1].grad,
        self.input_splits,
        self.output_splits,
        ep_group
    )
    if get_args().moe_zero_memory == 'level0':
        with torch.no_grad():
            recompute_fc1_input, _ = permute(perm_a2a_out, global_input_tokens_local_experts_indices)
            perm_a2a_out.untyped_storage().resize_(0)
            # restore fc1 input for dw computation
            dispached_input.untyped_storage().resize_(recompute_fc1_input.untyped_storage().size())
            dispached_input.untyped_storage().copy_(recompute_fc1_input.untyped_storage())
            recompute_fc1_input.untyped_storage().resize_(0)
    # dw computation
    if not in_detach_stage:
        WeightGradStore.pop()
    handle.wait()
    handle = None
    run_graph_backward(self.perm1_graph, perm1_out_grad)
    run_graph_backward(self.router_graph)
    run_graph_backward(self.pre_mlp_layernorm_graph)
    run_graph_backward(self.attn_graph)


    self.recompute_needed_tensors = [None for _ in range(len(self.recompute_needed_tensors))]

    return self.layer_input.grad


def transformer_layer_backward_dense(layer_output_grad, layer_graph):
    run_graph_backward(layer_graph.unperm2_graph, layer_output_grad)
    run_graph_backward(layer_graph.pre_mlp_layernorm_graph)
    run_graph_backward(layer_graph.attn_graph)

    return layer_graph.layer_input.grad


def transformer_layer_backward_noop(layer_output_grad, layer_graph):
    run_graph_backward(layer_graph.unperm2_graph, layer_output_grad, keep_grad=True)

    return layer_graph.layer_input.grad

