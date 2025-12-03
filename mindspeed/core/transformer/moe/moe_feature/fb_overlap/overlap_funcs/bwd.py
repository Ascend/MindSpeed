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
    dispatcher = self.layer.mlp.token_dispatcher
    use_shared_experts = hasattr(self.layer.mlp, 'shared_experts') and self.layer.mlp.shared_experts is not None

    dispached_input, probs, routing_map, bwd_num_global_tokens_per_local_expert_cpu = self.recompute_needed_tensors

    # Launch swap-in at the beginning of the backward pass.
    if self.unperm2_swap_manager:
        self.unperm2_swap_manager.async_swap_in(wait_stream=torch.npu.current_stream())
    if self.attn_swap_managers:
        for manager in self.attn_swap_managers:
            manager.async_swap_in(wait_stream=torch.npu.current_stream())

    if use_shared_experts:
        dispatcher.overlap_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(dispatcher.overlap_stream):
            shared_experts = self.layer.mlp.shared_experts
            shared_expert_grad = layer_output_grad if layer_output_grad is not None else self.unperm2_graph[1].grad
            shared_experts.pre_backward_comm(shared_expert_grad)

    run_graph_backward(self.unperm2_graph, layer_output_grad, keep_grad=True)

    if layer_output_grad is not None and not args.moe_unperm2_mem_optim_swap:
        layer_output_grad.untyped_storage().resize_(0)

    a2a_wait_event = shared_experts.pre_backward_handle if use_shared_experts else None
    # Do a synchronize for bwd a2a comm.
    torch.npu.current_stream().synchronize()
    unperm1_out_grad, handle = dispatcher.backward_async_combine_comm(
        self.unperm_a2a_graph[1].grad,
        input_splits=self.input_splits,
        output_splits=self.output_splits,
        output_splits_tp=self.output_splits_tp,
        wait_event=a2a_wait_event
    )
    # overlap alltoall by shared experts backward
    if use_shared_experts:
        with torch.npu.stream(dispatcher.overlap_stream):
            shared_experts.linear_fc2_act_fc1_backward(self.shared_experts_graph)

    if self.act_ckpt_manager is not None:
        self.act_ckpt_manager.recompute(True)
    handle.wait()

    # recomp permute1 and overlap all2all
    perm_a2a_handle = None
    if args.moe_zero_memory == 'level0':
        with torch.no_grad():
            input_before_perm1 = self.pre_mlp_layernorm_graph[0]

            def recomp_token_permutation1(hidden_states, routing_map):
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                permutated_local_input_tokens, _, _ = permute(
                    hidden_states, routing_map, num_out_tokens=dispatcher.num_out_tokens, fused=args.moe_permute_fusion
                )
                return permutated_local_input_tokens

            perm1_out = recomp_token_permutation1(input_before_perm1, routing_map)
            (perm_a2a_out, perm_a2a_handle), _ = dispatcher.async_dispatch_comm(
                perm1_out,
                input_splits=self.input_splits,
                output_splits=self.output_splits,
                output_splits_tp=self.output_splits_tp
            )

    if use_shared_experts:
        with torch.npu.stream(dispatcher.overlap_stream):
            shared_experts.post_backward_comm(wait_event=perm_a2a_handle)

    run_graph_backward(self.unperm1_graph, unperm1_out_grad)
    WeightGradStore.start_decouple()
    run_graph_backward(self.grouped_mlp_graph, keep_grad=True)  # keep for dw commputation
    if not in_detach_stage:
        WeightGradStore.end_decouple()
    run_graph_backward(self.perm2_graph, keep_graph=True)  # keep for dw commutation
    if args.moe_zero_memory == 'level0':
        perm_a2a_handle.wait()
        perm_a2a_handle = None

    if use_shared_experts:
        with torch.npu.stream(dispatcher.overlap_stream):
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

    if args.moe_zero_memory == 'level0':
        with torch.no_grad():
            recompute_fc1_input, _ = dispatcher.token_permute2(perm_a2a_out, None, bwd_num_global_tokens_per_local_expert_cpu)
            perm_a2a_out.untyped_storage().resize_(0)
            # restore fc1 input for dw computation
            dispached_input.untyped_storage().resize_(recompute_fc1_input.untyped_storage().size())
            dispached_input.untyped_storage().copy_(recompute_fc1_input.untyped_storage())
            recompute_fc1_input.untyped_storage().resize_(0)
    # dw computation
    if not in_detach_stage:
        WeightGradStore.pop()
    handle.wait()
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
    torch.npu.current_stream().wait_stream(dispatcher.overlap_stream)
    run_graph_backward(self.pre_mlp_layernorm_graph)
    if self.attn_swap_managers:
        for manager in self.attn_swap_managers:
            manager.wait_swap_in()
    run_graph_backward(self.attn_graph)

    self.recompute_needed_tensors = [None for _ in range(len(self.recompute_needed_tensors))]

    return getattr(self.layer_input, 'grad', None)


def transformer_layer_backward_dense(layer_output_grad, layer_graph):
    if layer_graph.attn_swap_managers:
        for manager in layer_graph.attn_swap_managers:
            manager.async_swap_in(wait_stream=torch.npu.current_stream())
    run_graph_backward(layer_graph.unperm2_graph, layer_output_grad)
    run_graph_backward(layer_graph.pre_mlp_layernorm_graph)
    if layer_graph.attn_swap_managers:
        for manager in layer_graph.attn_swap_managers:
            manager.wait_swap_in()
    run_graph_backward(layer_graph.attn_graph)

    return getattr(layer_graph.layer_input, 'grad', None)


def transformer_layer_backward_noop(layer_output_grad, layer_graph):
    run_graph_backward(layer_graph.unperm2_graph, layer_output_grad, keep_grad=True)

    return getattr(layer_graph.layer_input, 'grad', None)

