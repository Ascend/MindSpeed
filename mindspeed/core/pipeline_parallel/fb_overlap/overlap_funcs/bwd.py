#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import torch
import acl
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
    dispached_input, probs, indices, global_input_tokens_local_experts_indices = self.recompute_needed_tensors
    ep_group = parallel_state.get_expert_model_parallel_group()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()

    # Launch swap-in at the beginning of the backward pass.
    if self.unperm2_swap_manager:
        self.unperm2_swap_manager.async_swap_in(wait_stream=torch.npu.current_stream())

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
        # In case of unperm2 swap, layer_output_grad is required for probs_grad before router-backward
        if layer_output_grad is not None and not args.moe_unperm2_mem_optim_swap:
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
        self.act_ckpt_manager.recompute(True)
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
        self.perm_a2a_graph[1][0].grad,
        self.input_splits,
        self.output_splits,
        ep_group
    )

    perm1_prob_out_grad, prob_handle = None, None
    if args.moe_unperm2_mem_optim and '910B' not in acl.get_soc_name():
        _, perm1_prob_out_grad, prob_handle = async_all_to_all(
            self.perm_a2a_graph[1][1].grad,
            self.input_splits,
            self.output_splits,
            ep_group,
            event=handle,
            stream=torch.npu.current_stream()
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
    if prob_handle:
        prob_handle.wait()
    run_graph_backward(self.perm1_graph, [perm1_out_grad, perm1_prob_out_grad])

    # Swap-in unperm2 input for probs_grad computation in backward pass of router.
    if self.unperm2_swap_manager:
        self.unperm2_swap_manager.wait_swap_in()
    probs_grad = None
    if args.moe_unperm2_mem_optim_swap:
        # dprobs computation
        H = self.unperm2_swap_manager.npu_tensor.shape[-1]
        K = args.moe_router_topk
        probs_dtype = probs.dtype
        probs_grad = layer_output_grad.to(probs_dtype) * self.unperm2_swap_manager.npu_tensor.reshape(-1, K, H).to(probs_dtype)
        probs_grad = probs_grad.sum(dim=-1)
        layer_output_grad.untyped_storage().resize_(0)
        self.unperm2_swap_manager.npu_tensor.untyped_storage().resize_(0)
    run_graph_backward(self.router_graph, probs_grad)

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

