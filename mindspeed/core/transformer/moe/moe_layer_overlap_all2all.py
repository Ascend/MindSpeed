import torch

from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size
from megatron.core import tensor_parallel, parallel_state
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.training import get_args
from mindspeed.core.transformer.moe.comm_utils import async_all_to_all, async_all_gather
from mindspeed.core.transformer.moe.moe_utils import forward_func, backward_func
from mindspeed.ops.gmm import GMMFunction
from mindspeed.core.transformer.moe.moe_utils import AG_SHARED_EXPERTS_INPUTS, only_recompute_activation


def gmm_op(x, weight, bias, group_list, group_type):
    return GMMFunction.builder.load().npu_gmm([x], [weight], bias, group_list, group_type, 0)


class MoELayerOverlapAll2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, moe_layer: MoELayer):
        args = get_args()
        save_tensors = []
        save_tensors_for_grad = []
        ctx.input_shape = hidden_states.shape
        hidden_states = hidden_states.detach()
        hidden_states.requires_grad = True
        ctx.is_only_recompute_activation = only_recompute_activation(moe_layer.layer_number)

        # router
        with torch.enable_grad():
            scores, indices = moe_layer.router(hidden_states)

        save_tensors.append(scores)
        scores = scores.detach()
        scores.requires_grad = True
        save_tensors.append(scores)
        moe_zero_memory = args.moe_zero_memory
        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            ctx.activation_func = moe_layer.experts.activation_func
            ctx.hidden_size = moe_layer.experts.config.hidden_size
            ctx.num_local_experts = moe_layer.experts.num_local_experts
            ctx.weight1 = moe_layer.experts.weight1
            ctx.moe_grouped_gemm = moe_layer.token_dispatcher.config.moe_grouped_gemm
            ctx.num_local_experts = moe_layer.token_dispatcher.num_local_experts

        save_tensors.append(indices)

        if args.n_shared_experts:
            ctx.shared_experts = moe_layer.shared_experts
            if get_tensor_model_parallel_world_size() > 1:
                _, shared_experts_input, shared_experts_allgather_handle = async_all_gather(
                    hidden_states, get_tensor_model_parallel_group(), is_use_get_global_memory_buffer=True
                )
                AG_SHARED_EXPERTS_INPUTS.append((shared_experts_input, shared_experts_allgather_handle))
        else:
            ctx.shared_experts = None

        (share_experts_output, dispatched_input, tokens_per_expert) = moe_layer.token_dispatcher.token_permutation(
            hidden_states, scores, indices, ctx.shared_experts, save_tensors, save_tensors_for_grad, ctx
        )
        if isinstance(share_experts_output, tuple):
            share_experts_output, rs_share_experts_output, rs_shared_experts_handle = share_experts_output
        else:
            rs_share_experts_output = share_experts_output
            rs_shared_experts_handle = None
        (expert_output, mlp_bias), *_ = forward_func(moe_layer.experts, (dispatched_input, tokens_per_expert, ctx))
        save_tensors.append(expert_output)

        output, mlp_bias = moe_layer.token_dispatcher.token_unpermutation(expert_output, mlp_bias, save_tensors, save_tensors_for_grad)

        save_tensors.append(output)
        save_tensors.append(hidden_states)
        save_tensors_for_grad.append(hidden_states)

        save_tensors.append(share_experts_output)
        save_tensors.append(moe_layer.token_dispatcher.global_input_tokens_local_experts_indices)

        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            ctx.tokens_per_expert = tokens_per_expert

        ctx.save_for_backward(*save_tensors)
        ctx.output_splits = moe_layer.token_dispatcher.output_splits
        ctx.input_splits = moe_layer.token_dispatcher.input_splits
        ctx.router_topk = moe_layer.token_dispatcher.router_topk
        if args.n_shared_experts:
            if rs_shared_experts_handle is not None:
                rs_shared_experts_handle.wait()
            output_sum = output + rs_share_experts_output
            output.untyped_storage().resize_(0)
            share_experts_output.untyped_storage().resize_(0)
        else:
            output_sum = output.detach()

        ctx.saved_tensors_for_grad = save_tensors_for_grad
        return output_sum, mlp_bias

    @staticmethod
    def backward(ctx, *args):
        global_args = get_args()
        moe_zero_memory = global_args.moe_zero_memory
        (route_graph, detach_scores,
         indices,
         permute1_graph,
         permute2_input_detach, permute2_graph,
         experts_graph,
         unpermute1_input_detach, unpermute1_graph,
         unpermute2_input_detach, unpermute2_graph,
         detach_input, share_experts_graph,
         global_input_tokens_local_experts_indices,
         ) = ctx.saved_tensors

        ctx.save_for_backward()

        output_splits = ctx.output_splits
        input_splits = ctx.input_splits
        router_topk = ctx.router_topk

        from mindspeed.core.transformer.moe.moe_utils import set_gemm_backward_need_tensors
        set_gemm_backward_need_tensors(
            ((detach_input, indices, router_topk, global_input_tokens_local_experts_indices),
             permute2_input_detach, permute2_graph,
             output_splits, input_splits))

        if global_args.n_shared_experts:
            if get_tensor_model_parallel_world_size() > 1:
                _, backward_ag_shared, backward_ag_shared_handle = async_all_gather(
                    args[0], get_tensor_model_parallel_group()
                )
            else:
                backward_ag_shared = args[0]
                backward_ag_shared_handle = None

        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            with torch.no_grad():
                if get_tensor_model_parallel_world_size() > 1 and global_args.n_shared_experts:
                    _, shared_experts_input, shared_experts_allgather_handle = async_all_gather(
                        detach_input, get_tensor_model_parallel_group(), is_use_get_global_memory_buffer=True
                    )
                    AG_SHARED_EXPERTS_INPUTS.append((shared_experts_input, shared_experts_allgather_handle))

                # Recompute token rearrange in permutation1
                from megatron.core.transformer.moe.moe_utils import permute
                permutated_local_input_tokens, _ = permute(
                    detach_input.view(-1, detach_input.shape[-1]), indices, topk=ctx.router_topk
                )

                # Recompute expert parallel AlltoAll communication
                ep_group = parallel_state.get_expert_model_parallel_group()
                if global_args.moe_tp_extend_ep:
                    ep_group = parallel_state.get_tensor_and_expert_parallel_group()
                _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
                    permutated_local_input_tokens,
                    ctx.output_splits,
                    ctx.input_splits,
                    ep_group,
                )

        unpermute2_graph.backward(args[0])
        unpermute2_graph = None
        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            if global_args.n_shared_experts:
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

        ep_group = parallel_state.get_expert_model_parallel_group()
        if global_args.moe_tp_extend_ep:
            ep_group = parallel_state.get_tensor_and_expert_parallel_group()
        _, unpermute1_backward_input, handle = async_all_to_all(
            unpermute2_input_detach.grad,
            output_splits,
            input_splits,
            ep_group,
        )

        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            with torch.no_grad():
                if ctx.num_local_experts > 1:
                    # Recompute permutation2
                    global_input_tokens, _ = permute(
                        global_input_tokens, global_input_tokens_local_experts_indices
                    )
                    if not global_args.moe_tp_extend_ep and get_tensor_model_parallel_world_size() > 1 and ctx.moe_grouped_gemm:
                        global_input_tokens = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(
                            global_input_tokens
                        )
                # Recompute mm1 and act
                input_, mm1_out, act_out = ctx.recompute_tensors
                ctx.recompute_tensors = None
                if global_input_tokens.nelement() != 0:
                    group_list = torch.cumsum(ctx.tokens_per_expert, dim=0)
                    w1 = ctx.weight1.view(ctx.num_local_experts, ctx.hidden_size, -1)
                    mm1_out_ = gmm_op(global_input_tokens, w1, [], group_list, 0)[0]
                    group_list.untyped_storage().resize_(0)
                else:
                    w1 = ctx.weight1.view(ctx.hidden_size, -1)
                    mm1_out_ = torch.matmul(global_input_tokens, w1)

                act_out_ = ctx.activation_func(mm1_out_)
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
        handle.wait()
        unpermute2_input_detach.grad.untyped_storage().resize_(0)

        backward_func(unpermute1_graph, unpermute1_backward_input)

        unpermute1_backward_input.untyped_storage().resize_(0)

        backward_func(experts_graph, unpermute1_input_detach.grad)
        unpermute1_input_detach.grad.untyped_storage().resize_(0)

        from mindspeed.core.transformer.moe.moe_utils import get_all2all_experts_output
        permute1_backward_input, bw_permute1_ep_all2all_handle = get_all2all_experts_output()
        bw_permute1_ep_all2all_handle.wait()
        permute2_input_detach.grad.untyped_storage().resize_(0)
        backward_func(permute1_graph, permute1_backward_input)
        permute1_backward_input.untyped_storage().resize_(0)
        if moe_zero_memory != "disable":
            if ctx.router_topk > 1:
                from mindspeed.core.transformer.moe.moe_utils import get_prob_backward_need_tensors
                stream, matmul_output_grad, unpermuted_tokens = get_prob_backward_need_tensors()
                torch.npu.current_stream().wait_stream(stream)
                probs_grad = (matmul_output_grad * unpermuted_tokens).sum(-1).squeeze(-1)
                route_graph.backward(probs_grad)
            ctx.router_topk = None
        else:
            route_graph.backward(detach_scores.grad)
        route_graph = None
        grad_output = detach_input.grad
        return grad_output, None
