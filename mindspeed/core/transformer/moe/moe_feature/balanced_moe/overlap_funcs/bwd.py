# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from contextlib import AbstractContextManager

import torch
from megatron.core.transformer.moe.moe_utils import permute
from megatron.training import get_args

from mindspeed.core.transformer.moe.moe_feature.balanced_moe.communication import _groupedmlp_hot_expert_params_broadcast, \
    _groupedmlp_hot_expert_gradient_reduce
from mindspeed.core.transformer.moe.moe_feature.balanced_moe.modules.moe_layer import get_shared_params_for_hot_experts
from mindspeed.core.transformer.moe.moe_feature.balanced_moe.utils import CustomSliceFunction
from mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules.utils import run_graph_backward
from mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules.weight_grad_store import WeightGradStore


class NoDwDetachContext(AbstractContextManager):
    def __init__(self):
        pass

    def __enter__(self):
        # record state when enter the context
        # and disable dw detach
        self.orig_state = WeightGradStore.is_decoupleBlock
        WeightGradStore.end_decouple()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # recover dw detach state when exit the context
        WeightGradStore.is_decoupleBlock = self.orig_state


def recomp_token_permutation1(hidden_states, routing_map, sumnum_cold_local_hot_tokens):
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    permuted_input_tokens, _, _ = permute(
        hidden_states, routing_map
    )
    permuted_local_tokens, permuted_remote_hot_tokens = CustomSliceFunction.apply(
        permuted_input_tokens, sumnum_cold_local_hot_tokens)
    return permuted_local_tokens, permuted_remote_hot_tokens


def transformer_layer_backward_balanced_moe(
        layer_output_grad,
        layer_graph
):
    self = layer_graph
    args = get_args()
    in_detach_stage = WeightGradStore.is_decoupleBlock
    dispatcher = self.layer.mlp.token_dispatcher
    use_shared_experts = hasattr(self.layer.mlp, 'shared_experts') and self.layer.mlp.shared_experts is not None

    get_shared_params_for_hot_experts().register_shared_weight(self.layer.mlp.hot_experts)

    _groupedmlp_hot_expert_params_broadcast(
        self.layer.mlp.experts, self.hot_experts_list,
        self.layer.mlp.hot_experts, self.params,
    )

    cold_local_hot_input, remote_hot_input, probs, sorted_routing_map, \
        sumnum_cold_local_hot_tokens, bwd_num_global_tokens_per_local_expert_cpu = self.recompute_needed_tensors

    # Launch swap-in at the beginning of the backward pass.
    if self.unperm2_swap_manager:
        self.unperm2_swap_manager.async_swap_in(wait_stream=torch.npu.current_stream())
    if self.attn_swap_managers:
        for manager in self.attn_swap_managers:
            manager.async_swap_in(wait_stream=torch.npu.current_stream())

    if use_shared_experts:
        shared_experts = self.layer.mlp.shared_experts
        shared_expert_grad = layer_output_grad if layer_output_grad is not None else self.unperm2_graph[1].grad
        shared_experts.pre_backward_comm(shared_expert_grad)

    run_graph_backward(self.unperm2_graph, layer_output_grad, keep_grad=True)

    if layer_output_grad is not None and not args.moe_unperm2_mem_optim_swap:
        layer_output_grad.untyped_storage().resize_(0)

    a2a_wait_event = shared_experts.pre_backward_handle if use_shared_experts else None
    unperm1_out_grad, handle = dispatcher.backward_async_combine_comm(
        self.unperm_a2a_graph[1].grad,
        input_splits=self.input_splits,
        output_splits=self.output_splits,
        output_splits_tp=self.output_splits_tp,
        wait_event=a2a_wait_event
    )
    # overlap alltoall by shared experts backward
    if use_shared_experts:
        shared_experts.linear_fc2_act_fc1_backward(self.shared_experts_graph)

    if self.act_ckpt_manager is not None:
        self.act_ckpt_manager.recompute(True)
    if self.remote_hot_act_ckpt_manager is not None:
        self.remote_hot_act_ckpt_manager.recompute(True)
    handle.wait()

    # recomp permute1 and overlap all2all
    perm_a2a_handle = None
    if get_args().moe_zero_memory == 'level0':
        with torch.no_grad():
            input_before_perm1 = self.pre_mlp_layernorm_graph[0]

            perm1_local_out, perm1_remote_hot_out = recomp_token_permutation1(input_before_perm1,
                                                                              sorted_routing_map,
                                                                              sumnum_cold_local_hot_tokens)
            (perm_a2a_out, perm_a2a_handle), _ = dispatcher.async_dispatch_comm(
                perm1_local_out,
                input_splits=self.input_splits,
                output_splits=self.output_splits,
                output_splits_tp=self.output_splits_tp
            )

    if use_shared_experts:
        shared_experts.post_backward_comm(wait_event=perm_a2a_handle)

    run_graph_backward(self.unperm1_graph, unperm1_out_grad)
    WeightGradStore.start_decouple()
    local_experts_graph = (self.grouped_mlp_graph[0][0], self.grouped_mlp_graph[1][0])
    hot_experts_graph = (self.grouped_mlp_graph[0][1], self.grouped_mlp_graph[1][1])
    run_graph_backward(local_experts_graph, keep_grad=True)  # keep for dw commputation
    if not in_detach_stage:
        WeightGradStore.end_decouple()
    run_graph_backward(self.perm2_graph, keep_graph=True)  # keep for dw commutation
    if get_args().moe_zero_memory == 'level0':
        perm_a2a_handle.wait()
        perm_a2a_handle = None

    if use_shared_experts:
        shared_experts_grad = shared_experts.get_backward_grad()
        if shared_experts_grad is not None:
            self.pre_mlp_layernorm_graph[1].grad = shared_experts_grad

    (perm1_out_grad, handle), (perm1_prob_out_grad, prob_handle) = dispatcher.backward_async_dispatch_comm(
        self.perm_a2a_graph[1][0].grad,
        self.perm_a2a_graph[1][1].grad,
        input_splits=self.output_splits,
        output_splits=self.input_splits,
        input_splits_tp=self.output_splits_tp
    )

    if get_args().moe_zero_memory == 'level0':
        with torch.no_grad():
            recompute_fc1_input, _ = dispatcher.token_permute2(perm_a2a_out, None,
                                                               bwd_num_global_tokens_per_local_expert_cpu)
            perm_a2a_out.untyped_storage().resize_(0)
            # restore fc1 input for dw computation
            cold_local_hot_input.untyped_storage().resize_(recompute_fc1_input.untyped_storage().size())
            cold_local_hot_input.untyped_storage().copy_(recompute_fc1_input.untyped_storage())
            recompute_fc1_input.untyped_storage().resize_(0)

    hot_expert_broadcast_handles = self.params[4]
    for handles in hot_expert_broadcast_handles:
        for handle in handles:
            handle.wait()
        handles.clear()

    # hot experts dx + dw computation
    # here we do not do dw detach for hot experts; since dw for hot experts should do graident reduce comm latter.
    with NoDwDetachContext():
        run_graph_backward(hot_experts_graph)

    # dw computation
    if not in_detach_stage:
        WeightGradStore.pop()
    handle.wait()
    if prob_handle:
        prob_handle.wait()

    # hot experts gradient reduce comm is overlaped by attention backward.
    _groupedmlp_hot_expert_gradient_reduce(
        self.hot_experts_list, self.layer.mlp.hot_experts, self.params,
    )
    perm1_remote_hot_grad = self.perm1_graph[1][0].grad if self.perm1_graph[1][0] is not None else None
    perm1_remote_prob_grad = self.perm1_graph[1][1].grad if self.perm1_graph[1][1] is not None else None

    run_graph_backward(self.perm1_graph,
                       [perm1_out_grad, perm1_prob_out_grad, perm1_remote_hot_grad, perm1_remote_prob_grad])

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
    if self.attn_swap_managers:
        for manager in self.attn_swap_managers:
            manager.wait_swap_in()
    run_graph_backward(self.attn_graph)
    accum_hot_experts_grad(self.layer.mlp, self.hot_experts_list, self.hot_expert_inter_ep_grad_reduce_handles)

    self.recompute_needed_tensors = [None for _ in range(len(self.recompute_needed_tensors))]

    return getattr(self.layer_input, 'grad', None)


def accum_hot_experts_grad(mlp_layer, hot_experts_list, hot_expert_inter_ep_grad_reduce_handles):
    local_expert_indices = mlp_layer.local_expert_indices
    hot_expert_broadcast_handles = hot_expert_inter_ep_grad_reduce_handles[0]
    param, hot_expert_param = mlp_layer.experts.weight1, mlp_layer.hot_experts.weight1
    local_grad_view = param.main_grad.view(param.grad_local_shape)
    hot_grad_view = hot_expert_param.grad.view(param.grad_hot_shape)

    for param_offset, expert_id in enumerate(local_expert_indices):
        if expert_id in hot_experts_list:
            hot_expert_offset = hot_experts_list.index(expert_id)
            hot_expert_broadcast_handles[hot_expert_offset].wait()
            # Perform intra-EP local-hot expert gradient reduction.
            local_grad_view[param_offset].add_(hot_grad_view[hot_expert_offset])
    hot_expert_param.grad = None

    hot_expert_broadcast_handles = hot_expert_inter_ep_grad_reduce_handles[1]
    param, hot_expert_param = mlp_layer.experts.weight2, mlp_layer.hot_experts.weight2
    local_grad_view = param.main_grad.view(param.grad_local_shape)
    hot_grad_view = hot_expert_param.grad.view(param.grad_hot_shape)
    for param_offset, expert_id in enumerate(local_expert_indices):
        if expert_id in hot_experts_list:
            hot_expert_offset = hot_experts_list.index(expert_id)
            hot_expert_broadcast_handles[hot_expert_offset].wait()
            # Perform intra-EP local-hot expert gradient reduction.
            local_grad_view[param_offset].add_(hot_grad_view[hot_expert_offset])
    hot_expert_param.grad = None
