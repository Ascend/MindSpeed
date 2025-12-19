# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch

from mindspeed.core.transformer.moe.grouped_matmul_util import get_gmm_op_cls
from mindspeed.moe.utils import MoEAuxLossAutoScaler
from mindspeed.core.transformer.moe.moe_feature.overlap.comm_utils import async_all_to_all, async_all_gather
from mindspeed.ops.gmm import GMMFunction
from mindspeed.core.transformer.moe.moe_feature.overlap.moe_common import (
    forward_func, backward_func,
    AG_SHARED_EXPERTS_INPUTS, only_recompute_activation,
    set_gemm_backward_need_tensors, get_all2all_experts_output,
    )
from mindspeed.core.transformer.moe.moe_feature import (
    tensor_parallel,
    parallel_state,
    MoELayer,
    permute,
    save_to_aux_losses_tracker,
    sort_chunks_by_idxs)


class MoELayerOverlapAllToAllSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, config, moe_layer: MoELayer):
        ctx.config = config
        save_tensors = []
        ctx.input_shape = hidden_states.shape
        hidden_states = hidden_states.detach()
        hidden_states.requires_grad = True
        ctx.is_only_recompute_activation = only_recompute_activation(config, moe_layer.layer_number)
        moe_zero_memory = config.moe_zero_memory
        n_shared_experts = config.n_shared_experts
        ctx.moe_shared_expert_intermediate_size = config.moe_shared_expert_intermediate_size
        ctx.n_shared_experts = n_shared_experts
        ctx.moe_zero_memory = moe_zero_memory
        group_limited_greedy = hasattr(config, 'moe_router_load_balancing_type') and config.moe_router_load_balancing_type == "group_limited_greedy"

        # router
        with torch.enable_grad():
            scores, routing_map = moe_layer.router(hidden_states)

        save_tensors.append(scores)
        scores = scores.detach()
        scores.requires_grad = True
        save_tensors.append(scores)

        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            ctx.activation_func = moe_layer.experts.activation_func
            ctx.hidden_size = moe_layer.experts.config.hidden_size
            ctx.num_local_experts = moe_layer.experts.num_local_experts
            ctx.weight1 = moe_layer.experts.weight1
            ctx.moe_grouped_gemm = moe_layer.token_dispatcher.config.moe_grouped_gemm
            ctx.num_local_experts = moe_layer.token_dispatcher.num_local_experts

        save_tensors.append(routing_map)

        if n_shared_experts or ctx.moe_shared_expert_intermediate_size:
            ctx.shared_experts = moe_layer.shared_experts
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                _, shared_experts_input, shared_experts_allgather_handle = async_all_gather(
                    hidden_states, parallel_state.get_tensor_model_parallel_group(), is_use_get_global_memory_buffer=True
                )
                AG_SHARED_EXPERTS_INPUTS.append((shared_experts_input, shared_experts_allgather_handle))
        else:
            ctx.shared_experts = None

        (share_experts_output, dispatched_input, tokens_per_expert, global_probs) = moe_layer.token_dispatcher.token_permutation(
            hidden_states, scores, routing_map, ctx.shared_experts, save_tensors, ctx
        )

        if isinstance(share_experts_output, tuple):
            share_experts_output, rs_share_experts_output, rs_shared_experts_handle = share_experts_output
        else:
            if share_experts_output is not None:
                share_experts_output.requires_grad_(True)
            rs_share_experts_output = share_experts_output
            rs_shared_experts_handle = None
        (expert_output, mlp_bias), *_ = forward_func(moe_layer.experts, (dispatched_input, tokens_per_expert, global_probs, ctx))
        #experts_graph
        save_tensors.append(expert_output)
        output = moe_layer.token_dispatcher.token_unpermutation(expert_output, mlp_bias, save_tensors)

        if group_limited_greedy:
            save_tensors.append(moe_layer.router.l_aux)
            moe_layer.router.l_aux = moe_layer.router.l_aux.detach()
            moe_layer.router.l_aux.requires_grad = True
            save_tensors.append(moe_layer.router.l_aux)
            with torch.enable_grad():
                save_to_aux_losses_tracker(
                    "load_balancing_loss",
                    moe_layer.router.l_aux,
                    moe_layer.layer_number,
                    moe_layer.config.num_layers,
                )
                save_to_aux_losses_tracker(
                    "load_balancing_expert_level_loss",
                    moe_layer.router.l_expert_aux / config.moe_aux_loss_coeff,
                    moe_layer.layer_number,
                    moe_layer.config.num_layers,
                )
                if hasattr(moe_layer.router, 'l_device_aux'):
                    save_to_aux_losses_tracker(
                        "load_balancing_device_level_loss",
                        moe_layer.router.l_device_aux / config.moe_device_level_aux_loss_coeff,
                        moe_layer.layer_number,
                        moe_layer.config.num_layers,
                    )
                if hasattr(moe_layer.router, 'l_comm_aux'):
                    save_to_aux_losses_tracker(
                        "load_balancing_comm_level_loss",
                        moe_layer.router.l_comm_aux / config.moe_comm_aux_loss_coeff,
                        moe_layer.layer_number,
                        moe_layer.config.num_layers,
                    )
                output = MoEAuxLossAutoScaler.apply(output, moe_layer.router.l_aux)
        else:
            save_tensors.append(None)
            save_tensors.append(None)

        #unpermute2_graph
        save_tensors.append(output)
        #detach_input
        save_tensors.append(hidden_states)

        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            ctx.tokens_per_expert = tokens_per_expert

        ctx.output_splits = moe_layer.token_dispatcher.output_splits
        ctx.input_splits = moe_layer.token_dispatcher.input_splits
        ctx.router_topk = moe_layer.token_dispatcher.config.moe_router_topk
        ctx.num_out_tokens = moe_layer.token_dispatcher.num_out_tokens

        if n_shared_experts or ctx.moe_shared_expert_intermediate_size:
            if rs_shared_experts_handle is not None:
                rs_shared_experts_handle.wait()
            output_sum = output + rs_share_experts_output
            output.untyped_storage().resize_(0)
            rs_share_experts_output.untyped_storage().resize_(0)
            share_experts_output.untyped_storage().resize_(0)
        else:
            output_sum = output.detach()

        save_tensors.append(share_experts_output)
        if hasattr(moe_layer.token_dispatcher, 'global_input_tokens_local_experts_indices'):
            save_tensors.append(moe_layer.token_dispatcher.global_input_tokens_local_experts_indices)
        else:
            save_tensors.append(None)
        ctx.save_for_backward(*save_tensors)
        return output_sum, mlp_bias

    @staticmethod
    def backward(ctx, *args):
        config = ctx.config

        (route_graph, detach_scores,
         routing_map,
         permute1_graph, permuted_probs_graph,
         num_global_tokens_per_local_expert_cpu,
         permute2_input_detach, permute2_graph,
         permute2_prob_detach, permute2_prob_graph,
         experts_graph,
         unpermute1_input_detach, unpermute1_graph, unpermute2_input_detach,
          l_aux_graph, l_aux_detach, unpermute2_graph,
         detach_input, share_experts_graph,
         global_input_tokens_local_experts_indices,
         ) = ctx.saved_tensors

        n_shared_experts = ctx.n_shared_experts
        moe_zero_memory = ctx.moe_zero_memory
        moe_tp_extend_ep = config.moe_tp_extend_ep
        moe_shared_expert_intermediate_size = ctx.moe_shared_expert_intermediate_size

        output_splits = ctx.output_splits
        input_splits = ctx.input_splits
        num_out_tokens = ctx.num_out_tokens
        sort_input_by_local_experts = ctx.sort_input_by_local_experts

        if moe_tp_extend_ep:
            ep_group = parallel_state.get_expert_tensor_and_model_parallel_group()
        else:
            ep_group = parallel_state.get_expert_model_parallel_group()

        set_gemm_backward_need_tensors(
            ((detach_input, detach_scores, routing_map, num_global_tokens_per_local_expert_cpu, 
             sort_input_by_local_experts),
             permute2_input_detach, permute2_graph,
             permute2_prob_detach, permute2_prob_graph,
             output_splits, input_splits, num_out_tokens))

        if n_shared_experts:
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                _, backward_ag_shared, backward_ag_shared_handle = async_all_gather(
                    args[0], parallel_state.get_tensor_model_parallel_group()
                )
            else:
                backward_ag_shared = args[0]
                backward_ag_shared_handle = None

        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            with torch.no_grad():
                if parallel_state.get_tensor_model_parallel_world_size() > 1 and (n_shared_experts or moe_shared_expert_intermediate_size):
                    _, shared_experts_input, shared_experts_allgather_handle = async_all_gather(
                        detach_input, parallel_state.get_tensor_model_parallel_group(), is_use_get_global_memory_buffer=True
                    )
                    AG_SHARED_EXPERTS_INPUTS.append((shared_experts_input, shared_experts_allgather_handle))

                # Recompute token rearrange in permutation1
                permutated_local_input_tokens, permuted_probs, _ = permute(
                    detach_input.view(-1, detach_input.shape[-1]), routing_map, num_out_tokens=num_out_tokens, 
                    probs=detach_scores, fused=ctx.config.moe_permute_fusion
                )
                detach_scores.untyped_storage().resize_(0)

                # Recompute expert parallel and global_probs AlltoAll communication
                _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
                    permutated_local_input_tokens,
                    ctx.output_splits,
                    ctx.input_splits,
                    ep_group,
                )
                _, global_probs, permute1_probs_handle = async_all_to_all(
                    permuted_probs,
                    ctx.output_splits,
                    ctx.input_splits,
                    ep_group,
                )

        unpermute2_graph.backward(args[0])
        unpermute2_graph.untyped_storage().resize_(0)

        _, unpermute1_backward_input, unpermute1_handle = async_all_to_all(
            unpermute2_input_detach.grad,
            output_splits,
            input_splits,
            ep_group,
        )

        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            if n_shared_experts or moe_shared_expert_intermediate_size:
                with torch.no_grad():
                    # Recompute mm1 and act of shared experts
                    shared_fc1_out, bias_parallel = ctx.shared_experts.linear_fc1(detach_input)
                    shared_act_out = ctx.shared_experts.activation_function(shared_fc1_out, bias_parallel)
                    shared_act_out_size = shared_act_out.untyped_storage().size()
                    ctx.shared_act_out.untyped_storage().resize_(shared_act_out_size)
                    ctx.shared_act_out.untyped_storage().copy_(shared_act_out.untyped_storage())
                    shared_act_out.untyped_storage().resize_(0)
                    shared_fc1_out_size = shared_fc1_out.untyped_storage().size()
                    ctx.shared_fc1_out.untyped_storage().resize_(shared_fc1_out_size)
                    ctx.shared_fc1_out.untyped_storage().copy_(shared_fc1_out.untyped_storage())
                    shared_fc1_out.untyped_storage().resize_(0)
                if backward_ag_shared_handle is not None:
                    backward_ag_shared_handle.wait()
                share_experts_graph.backward(backward_ag_shared)
                share_experts_graph = None
                if backward_ag_shared_handle is not None:
                    backward_ag_shared.untyped_storage().resize_(0)
                ctx.shared_act_out.untyped_storage().resize_(0)
                ctx.shared_fc1_out.untyped_storage().resize_(0)

            permute1_ep_all_to_all_handle.wait()
            permutated_local_input_tokens.untyped_storage().resize_(0)

        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            with torch.no_grad():
                if ctx.num_local_experts > 1:
                    permute1_probs_handle.wait()
                    permuted_probs.untyped_storage().resize_(0)
                    # Recompute permutation2
                    global_input_tokens, permuted_probs_ = sort_chunks_by_idxs(
                        global_input_tokens,
                        num_global_tokens_per_local_expert_cpu.ravel(),
                        sort_input_by_local_experts,
                        probs=global_probs,
                    )
                    global_probs.untyped_storage().resize_(0)
                    if not moe_tp_extend_ep and parallel_state.get_expert_tensor_and_model_parallel_world_size() > 1 and ctx.moe_grouped_gemm:
                        global_input_tokens = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(
                            global_input_tokens
                        )
                # Recompute mm1 and act_with_probs.
                input_, mm1_out, expert_permuted_probs, act_out, act_without_probs = ctx.recompute_tensors
                ctx.recompute_tensors = None
                if global_input_tokens.nelement() != 0:
                    group_list = torch.cumsum(ctx.tokens_per_expert, dim=0)
                    w1 = ctx.weight1.view(ctx.num_local_experts, ctx.hidden_size, -1)
                    mm1_out_ = get_gmm_op_cls().op_forward(global_input_tokens, w1, group_list)[0]
                    group_list.untyped_storage().resize_(0)
                else:
                    w1 = ctx.weight1.view(ctx.hidden_size, -1)
                    mm1_out_ = torch.matmul(global_input_tokens, w1)

                act_without_probs_ = ctx.activation_func(mm1_out_)
                act_out_ = act_without_probs_ * permuted_probs_.unsqueeze(-1)

                act_without_probs_size = act_without_probs_.untyped_storage().size()
                act_without_probs.untyped_storage().resize_(act_without_probs_size)
                act_without_probs.untyped_storage().copy_(act_without_probs_.untyped_storage())
                act_without_probs = None
                act_without_probs_.untyped_storage().resize_(0)
                act_out_size = act_out_.untyped_storage().size()
                act_out.untyped_storage().resize_(act_out_size)
                act_out.untyped_storage().copy_(act_out_.untyped_storage())
                act_out = None
                act_out_.untyped_storage().resize_(0)
                mm1_out_size = mm1_out_.untyped_storage().size()
                mm1_out.untyped_storage().resize_(mm1_out_size)
                mm1_out.untyped_storage().copy_(mm1_out_.untyped_storage())
                mm1_out = None
                mm1_out_.untyped_storage().resize_(0)
                permuted_probs_size = permuted_probs_.untyped_storage().size()
                expert_permuted_probs.untyped_storage().resize_(permuted_probs_size)
                expert_permuted_probs.untyped_storage().copy_(permuted_probs_.untyped_storage())
                expert_permuted_probs = None
                permuted_probs_.untyped_storage().resize_(0)
                input_size = global_input_tokens.untyped_storage().size()
                input_.untyped_storage().resize_(input_size)
                input_.untyped_storage().copy_(global_input_tokens.untyped_storage())
                input_ = None
                global_input_tokens.untyped_storage().resize_(0)
            ctx.activation_func = None
            ctx.hidden_size = None
            ctx.num_local_experts = None
            ctx.weight1 = None
            ctx.moe_grouped_gemm = None
            ctx.num_local_experts = None
            ctx.input_splits = None
            ctx.output_splits = None
        elif share_experts_graph is not None:
            if backward_ag_shared_handle is not None:
                backward_ag_shared_handle.wait()
            share_experts_graph.backward(backward_ag_shared)
            share_experts_graph = None
            if backward_ag_shared_handle is not None:
                backward_ag_shared.untyped_storage().resize_(0)

        if unpermute1_handle is not None:
            unpermute1_handle.wait()
            unpermute2_input_detach.grad.untyped_storage().resize_(0)

        backward_func(unpermute1_graph, unpermute1_backward_input)
        unpermute1_backward_input.untyped_storage().resize_(0)

        backward_func(experts_graph, unpermute1_input_detach.grad)
        unpermute1_input_detach.grad.untyped_storage().resize_(0)

        (permute1_backward_input, bw_permute1_ep_all2all_handle, 
        permute1_prob_backward_input, bw_permute1_prob_all2all_handle) = get_all2all_experts_output()

        bw_permute1_prob_all2all_handle.wait()

        bw_permute1_ep_all2all_handle.wait()
        torch.autograd.backward([permute1_graph, permuted_probs_graph],
                                grad_tensors=[permute1_backward_input, permute1_prob_backward_input])
        permute1_backward_input.untyped_storage().resize_(0)
        permute1_prob_backward_input.untyped_storage().resize_(0)

        if l_aux_graph is not None:
            l_aux_graph.backward(l_aux_detach.grad, retain_graph=True)
        route_graph.backward(detach_scores.grad)
        route_graph = None
        grad_output = detach_input.grad
        return grad_output, None, None
