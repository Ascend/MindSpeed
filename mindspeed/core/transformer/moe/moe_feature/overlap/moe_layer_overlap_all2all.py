# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import inspect
import torch
from mindspeed.moe.utils import MoEAuxLossAutoScaler
from mindspeed.core.transformer.moe.moe_feature.overlap.comm_utils import async_all_to_all
from mindspeed.ops.gmm import GMMFunction
from mindspeed.core.transformer.moe.moe_feature.overlap.moe_common import (
    forward_func, backward_func,
    only_recompute_activation,
    set_gemm_backward_need_tensors, get_all2all_experts_output,
    )
from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu
from mindspeed.core.transformer.moe.moe_feature.overlap.moe_common import only_recompute_activation
from mindspeed.core.transformer.moe.moe_feature import (
    tensor_parallel,
    parallel_state,
    MoELayer,
    permute,
    save_to_aux_losses_tracker,
    sort_chunks_by_idxs,
    gather_from_sequence_parallel_region,
    )


def gmm_op(x, weight, bias, group_list, group_type):
    return GMMFunction.builder.load().npu_gmm([x], [weight], bias, group_list, group_type, 0)


class MoELayerOverlapAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, config, moe_layer: MoELayer):
        ctx.config = config
        save_tensors = []
        ctx.input_shape = hidden_states.shape
        ctx.moe_layer = moe_layer
        hidden_states = hidden_states.detach()
        hidden_states.requires_grad = True
        ctx.is_only_recompute_activation = only_recompute_activation(config, moe_layer.layer_number)
        # router
        with torch.enable_grad():
            scores, routing_map = moe_layer.router(hidden_states)

        save_tensors.append(scores)
        scores = scores.detach()
        scores.requires_grad = True
        save_tensors.append(scores)
        moe_zero_memory = config.moe_zero_memory
        n_shared_experts = config.n_shared_experts
        ctx.n_shared_experts = n_shared_experts
        ctx.moe_zero_memory = moe_zero_memory
        moe_shared_expert_intermediate_size = config.moe_shared_expert_intermediate_size
        group_limited_greedy = hasattr(config, 'moe_router_load_balancing_type') and config.moe_router_load_balancing_type == "group_limited_greedy"
        ctx.shared_expert_overlap = moe_layer.shared_expert_overlap

        # if shared_expert_overlap, save share_experts graph separately for backward.
        if ctx.shared_expert_overlap:
            ctx.share_experts_graph_list = []
        else:
            ctx.share_experts_graph_list = None

        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            ctx.activation_func_with_probs = moe_layer.experts.activation_func_with_probs
            ctx.hidden_size = moe_layer.experts.config.hidden_size
            ctx.num_local_experts = moe_layer.experts.num_local_experts
            ctx.weight1 = moe_layer.experts.weight1
            ctx.moe_grouped_gemm = moe_layer.token_dispatcher.config.moe_grouped_gemm
            ctx.num_local_experts = moe_layer.token_dispatcher.num_local_experts

        if n_shared_experts or moe_shared_expert_intermediate_size:
            ctx.shared_experts = moe_layer.shared_experts
            if config.moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
                moe_layer.shared_experts.activation_function = fused_swiglu if config.gated_linear_unit else moe_layer.shared_experts.activation_func

        save_tensors.append(routing_map)

        (dispatched_input, tokens_per_expert, global_probs) = moe_layer.token_dispatcher.token_permutation(
            hidden_states, scores, routing_map, save_tensors, ctx
        )

        (expert_output, mlp_bias), *_ = forward_func(moe_layer.experts, (dispatched_input, tokens_per_expert, global_probs, ctx))
        save_tensors.append(expert_output)
        (output), expert_output_datach, *_ = forward_func(moe_layer.token_dispatcher.token_unpermutation, (expert_output, mlp_bias, ctx))
        #unpermute1_input_detach
        save_tensors.append(expert_output_datach)
        expert_output_datach.untyped_storage().resize_(0)

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
        ctx.num_global_tokens_per_local_expert_cpu = moe_layer.token_dispatcher.num_global_tokens_per_local_expert_cpu
        ctx.sort_input_by_local_experts = moe_layer.token_dispatcher.sort_input_by_local_experts
        ctx.output_splits_tp = moe_layer.token_dispatcher.output_splits_tp

        #save shared_experts overlap backwards tensor.
        if moe_layer.shared_expert_overlap:
            ctx.save_for_backward(*ctx.share_experts_graph_list)

        output_sum = output.detach()
        ctx.save_for_backward(*save_tensors)
        return output_sum, mlp_bias

    @staticmethod
    def backward(ctx, *args):

        (route_graph, detach_scores,
         routing_map,
         permute1_graph, permuted_probs_graph,
         permute2_input_detach, permute2_graph,
         permute2_prob_detach, permute2_prob_graph,
         experts_graph,
         unpermute1_input_detach,
         l_aux_graph, l_aux_detach, unpermute2_graph,
         detach_input, 
         ) = ctx.saved_tensors

        n_shared_experts = ctx.n_shared_experts
        moe_zero_memory = ctx.moe_zero_memory

        output_splits = ctx.output_splits
        input_splits = ctx.input_splits

        num_global_tokens_per_local_expert_cpu = ctx.num_global_tokens_per_local_expert_cpu
        sort_input_by_local_experts = ctx.sort_input_by_local_experts
        output_splits_tp = ctx.output_splits_tp

        #Get share_expert_graph for backward.
        if ctx.moe_layer.shared_expert_overlap:
            (cached_fc1_input_graph, 
            cached_fc1_input_detach,
        ) = ctx.share_experts_graph_list

        set_gemm_backward_need_tensors(
            ((detach_input, detach_scores, routing_map, num_global_tokens_per_local_expert_cpu, 
             sort_input_by_local_experts),
             permute2_input_detach, permute2_graph,
             permute2_prob_detach, permute2_prob_graph,
             output_splits, input_splits, output_splits_tp))

        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            with torch.no_grad():
                # Recompute token rearrange in permutation1
                permutated_local_input_tokens, permuted_probs, _ = permute(
                    detach_input.view(-1, detach_input.shape[-1]), routing_map, probs=detach_scores,
                )

                if parallel_state.get_expert_tensor_parallel_world_size() > 1:
                    cached_fc1_input_ = gather_from_sequence_parallel_region(
                        detach_input, tensor_parallel_output_grad=True
                    )
                else:
                    cached_fc1_input_ = detach_input

                # Recompute expert parallel AlltoAll communication.
                _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
                    permutated_local_input_tokens,
                    ctx.output_splits,
                    ctx.input_splits,
                    parallel_state.get_expert_model_parallel_group(),
                )
                # Recompute cached_fc1_input.
                # Use Megatron's shared experts fc1 to overlap global_input_tokens Alltoall.
                if n_shared_experts:
                    cached_fc1_input_size = cached_fc1_input_.untyped_storage().size()
                    ctx.cached_fc1_input.untyped_storage().resize_(cached_fc1_input_size)
                    ctx.cached_fc1_input.untyped_storage().copy_(cached_fc1_input_.untyped_storage())
                    shared_fc1_out, bias_parallel = ctx.moe_layer.token_dispatcher.shared_experts.linear_fc1(cached_fc1_input_)
                    #Avoid cached_fc1_input memory blast when TP=1.
                    if parallel_state.get_expert_tensor_parallel_world_size() > 1:
                        cached_fc1_input_.untyped_storage().resize_(0)

                permute1_ep_all_to_all_handle.wait()
                permutated_local_input_tokens.untyped_storage().resize_(0)

                _, global_probs, permute1_probs_handle = async_all_to_all(
                    permuted_probs,
                    ctx.output_splits,
                    ctx.input_splits,
                    parallel_state.get_expert_model_parallel_group(),
                )

        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            if n_shared_experts:
                with torch.no_grad():
                    # Recompute act of shared experts.
                    # Use Megatron's shared experts act to overlap global_probs Alltoall.
                    from mindspeed.core.transformer.moe.moe_feature import (
                        bias_geglu_impl,
                        bias_gelu_impl,
                        bias_swiglu_impl)
                    import torch.nn.functional as F
                    if ctx.moe_layer.token_dispatcher.shared_experts.config.bias_activation_fusion:
                        if ctx.moe_layer.token_dispatcher.shared_experts.activation_func == F.gelu:
                            if ctx.moe_layer.token_dispatcher.shared_experts.config.gated_linear_unit:
                                shared_act_out = bias_geglu_impl(
                                    shared_fc1_out, bias_parallel
                                )
                            else:
                                assert ctx.moe_layer.token_dispatcher.shared_experts.config.add_bias_linear is True
                                shared_act_out = bias_gelu_impl(shared_fc1_out, bias_parallel)
                        elif ctx.moe_layer.token_dispatcher.shared_experts.activation_func == F.silu and ctx.moe_layer.token_dispatcher.shared_experts.config.gated_linear_unit:
                            shared_act_out = bias_swiglu_impl(
                                shared_fc1_out,
                                bias_parallel,
                                ctx.moe_layer.token_dispatcher.shared_experts.config.activation_func_fp8_input_store,
                            )
                        else:
                            raise ValueError("Only support fusion of gelu and swiglu")
                    else:
                        if bias_parallel is not None:
                            shared_act_out = shared_fc1_out + bias_parallel
                        if ctx.moe_layer.token_dispatcher.shared_experts.config.gated_linear_unit:

                            def glu(x):
                                x = torch.chunk(x, 2, dim=-1)
                                return ctx.moe_layer.token_dispatcher.shared_experts.config.activation_func(x[0]) * x[1]

                            shared_act_out = glu(shared_fc1_out)
                        else:
                            shared_act_out = ctx.moe_layer.token_dispatcher.shared_experts.activation_func(shared_fc1_out)

                    shared_act_out_size = shared_act_out.untyped_storage().size()
                    ctx.shared_act_out.untyped_storage().resize_(shared_act_out_size)
                    ctx.shared_act_out.untyped_storage().copy_(shared_act_out.untyped_storage())
                    shared_act_out.untyped_storage().resize_(0)

                    shared_fc1_out_size = shared_fc1_out.untyped_storage().size()
                    ctx.cached_fc1_output.untyped_storage().resize_(shared_fc1_out_size)
                    ctx.cached_fc1_output.untyped_storage().copy_(shared_fc1_out.untyped_storage())
                    shared_fc1_out.untyped_storage().resize_(0)

        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            with torch.no_grad():
                if parallel_state.get_expert_tensor_parallel_world_size() > 1:
                    global_input_tokens = tensor_parallel.gather_from_sequence_parallel_region(
                        global_input_tokens,
                        group=parallel_state.get_expert_tensor_parallel_group(),
                        output_split_sizes=(
                            output_splits_tp.tolist() if output_splits_tp is not None else None
                        ),
                    )

                permute1_probs_handle.wait()
                if parallel_state.get_expert_tensor_parallel_world_size() > 1:
                    global_probs = tensor_parallel.gather_from_sequence_parallel_region(
                        global_probs,
                        group=parallel_state.get_expert_tensor_parallel_group(),
                        output_split_sizes=(
                            output_splits_tp.tolist() if output_splits_tp is not None else None
                        ),
                    )

                if ctx.num_local_experts > 1:
                    permuted_probs.untyped_storage().resize_(0)
                    # Recompute permutation2.
                    global_input_tokens, permuted_probs_ = sort_chunks_by_idxs(
                        global_input_tokens,
                        num_global_tokens_per_local_expert_cpu.ravel(),
                        sort_input_by_local_experts,
                        probs=global_probs,
                    )
                global_probs.untyped_storage().resize_(0)
                # Recompute mm1 and act.
                input_, mm1_out, permuted_probs, act_out = ctx.recompute_tensors
                ctx.recompute_tensors = None
                if global_input_tokens.nelement() != 0:
                    group_list = torch.cumsum(ctx.tokens_per_expert, dim=0)
                    w1 = ctx.weight1.view(ctx.num_local_experts, ctx.hidden_size, -1)
                    mm1_out_ = gmm_op(global_input_tokens, w1, [], group_list, 0)[0]
                    group_list.untyped_storage().resize_(0)
                else:
                    w1 = ctx.weight1.view(ctx.hidden_size, -1)
                    mm1_out_ = torch.matmul(global_input_tokens, w1)

                act_out_ = ctx.activation_func_with_probs(mm1_out_, permuted_probs_.unsqueeze(-1))
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
                permuted_probs.untyped_storage().resize_(permuted_probs_size)
                permuted_probs.untyped_storage().copy_(permuted_probs_.untyped_storage())
                permuted_probs = None
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

        # unpermute backward.
        unpermute2_graph.backward(args[0])
        unpermute2_graph = None

        backward_func(experts_graph, unpermute1_input_detach.grad)
        unpermute1_input_detach.grad.untyped_storage().resize_(0)

        (permute1_backward_input, bw_permute1_ep_all2all_handle, 
        permute1_prob_backward_input, bw_permute1_prob_all2all_handle) = get_all2all_experts_output()

        #Overlap with async alltoall from GeMM's backward.
        if n_shared_experts:
            with torch.cuda.stream(ctx.moe_layer.shared_experts.stream):
                backward_func(cached_fc1_input_graph, cached_fc1_input_detach.grad) 
                #Avoid cached_fc1_input memory blast when TP=1.
                if parallel_state.get_expert_tensor_parallel_world_size() > 1:
                    cached_fc1_input_graph.untyped_storage().resize_(0)
                    cached_fc1_input_detach.grad.untyped_storage().resize_(0)

        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            if n_shared_experts:
                ctx.shared_act_out.untyped_storage().resize_(0)
                ctx.cached_fc1_output.untyped_storage().resize_(0)

        bw_permute1_ep_all2all_handle.wait()
        permute2_input_detach.grad.untyped_storage().resize_(0)

        bw_permute1_prob_all2all_handle.wait()
        # permute1_graph and permuted_probs_graph are in the same graph, do not execute the backward_func twice
        torch.autograd.backward([permute1_graph, permuted_probs_graph],
                                grad_tensors=[permute1_backward_input, permute1_prob_backward_input])
        permute1_backward_input.untyped_storage().resize_(0)
        permute1_prob_backward_input.untyped_storage().resize_(0)

        if l_aux_graph is not None:
            l_aux_graph.backward(l_aux_detach.grad, retain_graph=True)

        route_graph.backward(detach_scores.grad)
        route_graph = None
        grad_output = detach_input.grad
        #Wait stream for share_expert_overlap.
        if n_shared_experts:
            ctx.moe_layer.shared_experts.stream.wait_stream(torch.cuda.current_stream())
        return grad_output, None, None