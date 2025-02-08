# Copyright (c) 2024, Huawei Technologies.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size
from megatron.core import tensor_parallel, parallel_state
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.training import get_args
from megatron.core.transformer.moe.moe_utils import permute, save_to_aux_losses_tracker
from mindspeed.moe.utils import MoEAuxLossAutoScaler
from mindspeed.core.transformer.moe.comm_utils import (async_all_to_all, async_all_gather, async_reduce_scatter,
                                                       transfer_tensor_last_dim_to_first, transfer_tensor_first_dim_to_last)
from mindspeed.core.transformer.moe.moe_utils import (forward_func, backward_func, permute_with_ep)
from mindspeed.ops.gmm import GMMFunction
from mindspeed.core.transformer.moe.moe_utils import (AG_SHARED_EXPERTS_INPUTS, only_recompute_activation,
                                                      set_gemm_backward_need_tensors,
                                                      set_all2all_experts_output, get_all2all_experts_output,
                                                      get_prob_backward_need_tensors,
                                                      set_permute_with_ep_local_input_tokens)


def gmm_op(x, weight, bias, group_list, group_type):
    if isinstance(group_list, torch.Tensor) and group_list.device.type == 'cpu':
        group_list = group_list.tolist()
    return GMMFunction.builder.load().npu_gmm([x], [weight], bias, group_list, group_type, 0)


def moe_experts_pipeline_forward_func(tokens_per_expert, moe_layer, dispatched_input, ctx, save_tensors):
    input_list = []
    expert_graphs = []
    expert_outputs = []
    tokens_per_expert_list = []
    moe_experts_pipeline_degree = ctx.moe_experts_pipeline_degree

    # 1. 划分子集
    # 赋值self.input_list和self.tokens_per_expert_list
    tokens_per_expert = tokens_per_expert.cpu()
    group_list = torch.cumsum(tokens_per_expert, dim=0)
    num_experts_overlap = moe_layer.num_local_experts // moe_experts_pipeline_degree

    for i in range(moe_experts_pipeline_degree):
        start_id = i * num_experts_overlap
        start = 0
        if i != 0:
            start = group_list[start_id - 1]
        end_id = (i + 1) * num_experts_overlap
        end = group_list[end_id - 1]
        input_i = dispatched_input[start : end]
        tokens_per_expert_i = tokens_per_expert[start_id : end_id]
        input_list.append(input_i)
        tokens_per_expert_list.append(tokens_per_expert_i)
    ctx.input_list = input_list

    # 2. 对每个专家子集的输入数据进行模型计算，并将计算结果保存在expert_outputs中
    ag_handle_i_next = None
    rs_handle_i = None
    input_i_next = None
    num_dim = None
    rs_input_i = None

    for i in range(moe_experts_pipeline_degree):
        if i == 0:
            _, input_i, ag_handle_i = async_all_gather(input_list[i], get_tensor_model_parallel_group(), last_dim=True)
            _, input_i_next, ag_handle_i_next = async_all_gather(input_list[i + 1], get_tensor_model_parallel_group(), last_dim=True)
        elif i != (moe_experts_pipeline_degree - 1):
            input_i = input_i_next
            ag_handle_i = ag_handle_i_next
            _, input_i_next, ag_handle_i_next = async_all_gather(input_list[i + 1], get_tensor_model_parallel_group(),
                                                                 last_dim=True)
        else:
            input_i = input_i_next
            ag_handle_i = ag_handle_i_next

        ag_handle_i.wait()
        input_i = torch.cat(input_i, dim=input_list[i].dim() - 1).contiguous()
        input_i = input_i.detach()
        input_i.requires_grad = True
        (expert_output, mlp_bias), *_ = forward_func(moe_layer.experts[i], (input_i, tokens_per_expert_list[i], ctx))
        if rs_handle_i is not None:
            rs_handle_i.wait()
            rs_input_i.untyped_storage().resize_(0)
            expert_graphs[i - 1].untyped_storage().resize_(0)
            expert_outputs[i - 1] = transfer_tensor_first_dim_to_last(expert_outputs[i - 1], num_dim)
            expert_outputs[i - 1].requires_grad = True
        # sub expert graph
        expert_graphs.append(expert_output)

        expert_output, num_dim = transfer_tensor_last_dim_to_first(expert_output)
        rs_input_i, rs_expert_output, rs_handle_i = async_reduce_scatter(expert_output, get_tensor_model_parallel_group())

        expert_outputs.append(rs_expert_output)

        if i == (moe_experts_pipeline_degree - 1):
            rs_handle_i.wait()
            rs_input_i.untyped_storage().resize_(0)
            expert_graphs[i].untyped_storage().resize_(0)
            expert_outputs[i] = transfer_tensor_first_dim_to_last(expert_outputs[i], num_dim)
            expert_outputs[i].requires_grad = True

    ctx.expert_graphs = expert_graphs
    ctx.expert_outputs = expert_outputs

    # 3. 将所有子集的计算结果拼接在一起，保存在`expert_output`中
    with torch.enable_grad():
        expert_output = torch.cat(expert_outputs, dim=0)

    for temp in expert_outputs:
        temp.untyped_storage().resize_(0)

    return expert_output, mlp_bias


def moe_experts_pipeline_backward_func(ctx, input_list):
    expert_grad_outputs = []

    ag_handle_i_next = None
    rs_handle_i = None
    input_i_next = None
    num_dim = None
    mm1_inputs_grad = None
    ag_input_i = None
    ag_input_i_next = None
    rs_input_i = None
    ag_input_list = []

    moe_experts_pipeline_degree = ctx.moe_experts_pipeline_degree
    expert_graphs = ctx. expert_graphs
    expert_outputs = ctx.expert_outputs

    for i in range(moe_experts_pipeline_degree):
        if i == 0:
            ag_input_i, input_i, ag_handle_i = async_all_gather(expert_outputs[i].grad, get_tensor_model_parallel_group(),
                                                       last_dim=True)
            ag_input_i_next, input_i_next, ag_handle_i_next = async_all_gather(expert_outputs[i + 1].grad,
                                                                 get_tensor_model_parallel_group(),
                                                                 last_dim=True)
        elif i != (moe_experts_pipeline_degree - 1):
            input_i = input_i_next
            ag_handle_i = ag_handle_i_next
            ag_input_i = ag_input_i_next
            ag_input_i_next, input_i_next, ag_handle_i_next = async_all_gather(expert_outputs[i + 1].grad,
                                                                 get_tensor_model_parallel_group(),
                                                                 last_dim=True)
        else:
            input_i = input_i_next
            ag_handle_i = ag_handle_i_next
            ag_input_i = ag_input_i_next

        ag_handle_i.wait()
        ag_input_list.append(ag_input_i)
        input_i = torch.cat(input_i, dim=expert_outputs[i].grad.dim() - 1).contiguous()

        set_gemm_backward_need_tensors(input_list[i])

        backward_func(expert_graphs[i], input_i)

        if rs_handle_i is not None:
            rs_handle_i.wait()
            rs_input_i.untyped_storage().resize_(0)
            mm1_inputs_grad.untyped_storage().resize_(0)
            expert_grad_outputs[i - 1] = transfer_tensor_first_dim_to_last(expert_grad_outputs[i - 1], num_dim)

        rs_input_i, expert_output, rs_handle_i, mm1_inputs_grad, num_dim = get_all2all_experts_output()
        expert_grad_outputs.append(expert_output)

        if i == (moe_experts_pipeline_degree - 1):
            rs_handle_i.wait()
            rs_input_i.untyped_storage().resize_(0)
            mm1_inputs_grad.untyped_storage().resize_(0)
            expert_grad_outputs[i] = transfer_tensor_first_dim_to_last(expert_grad_outputs[i], num_dim)

    for ag_input in ag_input_list:
        ag_input.untyped_storage().resize_(0)

    expert_grad_output = torch.cat(expert_grad_outputs, dim=0)
    return expert_grad_output


class MoELayerOverlapAll2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, moe_layer: MoELayer):
        args = get_args()
        moe_hierarchical_alltoallv = args.moe_hierarchical_alltoallv
        moe_experts_pipeline_degree = args.moe_experts_pipeline_degree
        ctx.moe_experts_pipeline_degree = moe_experts_pipeline_degree
        save_tensors = []
        ctx.input_shape = hidden_states.shape
        hidden_states = hidden_states.detach()
        hidden_states.requires_grad = True
        ctx.is_only_recompute_activation = only_recompute_activation(moe_layer.layer_number)
        ctx.layer_number = moe_layer.layer_number
        if not moe_hierarchical_alltoallv and args.n_shared_experts:
            if get_tensor_model_parallel_world_size() > 1:
                _, shared_experts_input, shared_experts_allgather_handle = async_all_gather(
                    hidden_states, get_tensor_model_parallel_group(), is_use_get_global_memory_buffer=True
                )
                AG_SHARED_EXPERTS_INPUTS.append((shared_experts_input, shared_experts_allgather_handle))

        # router
        with torch.enable_grad():
            scores, indices = moe_layer.router(hidden_states)

        save_tensors.append(scores)
        scores = scores.detach()
        scores.requires_grad = True
        save_tensors.append(scores)
        moe_zero_memory = args.moe_zero_memory
        n_shared_experts = args.n_shared_experts
        ctx.n_shared_experts = n_shared_experts
        ctx.moe_zero_memory = moe_zero_memory
        shared_expert_gate = hasattr(args, 'shared_expert_gate') and args.shared_expert_gate
        group_limited_greedy = hasattr(args, 'moe_router_load_balancing_type') and args.moe_router_load_balancing_type == "group_limited_greedy"
        ctx.shared_expert_gate = shared_expert_gate

        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            ctx.activation_func = moe_layer.experts.activation_func
            ctx.hidden_size = moe_layer.experts.config.hidden_size
            ctx.num_local_experts = moe_layer.experts.num_local_experts
            ctx.weight1 = moe_layer.experts.weight1
            ctx.moe_grouped_gemm = moe_layer.token_dispatcher.config.moe_grouped_gemm
            ctx.num_local_experts = moe_layer.token_dispatcher.num_local_experts

        save_tensors.append(indices)

        if n_shared_experts:
            ctx.shared_experts = moe_layer.shared_experts
        else:
            ctx.shared_experts = None

        if shared_expert_gate:
            shared_expert_gate = moe_layer.shared_expert_gate
        else:
            shared_expert_gate = None

        (share_experts_output, dispatched_input, tokens_per_expert) = moe_layer.token_dispatcher.token_permutation(
            hidden_states, scores, indices, ctx.shared_experts, save_tensors, shared_expert_gate, ctx
        )
        if moe_experts_pipeline_degree:
            save_tensors.append(None)
            save_tensors.append(None)
            expert_output, mlp_bias = moe_experts_pipeline_forward_func(tokens_per_expert, moe_layer, dispatched_input, ctx, save_tensors)
            output, mlp_bias = moe_layer.token_dispatcher.token_unpermutation(expert_output, mlp_bias, save_tensors)


            if isinstance(share_experts_output, tuple):
                share_experts_output, rs_share_experts_output, rs_shared_experts_handle = share_experts_output
            else:
                rs_share_experts_output = share_experts_output
                rs_shared_experts_handle = None

            expert_output.untyped_storage().resize_(0)
        else:

            if isinstance(share_experts_output, tuple):
                share_experts_output, rs_share_experts_output, rs_shared_experts_handle = share_experts_output
            else:
                rs_share_experts_output = share_experts_output
                rs_shared_experts_handle = None

            (expert_output, mlp_bias), *_ = forward_func(moe_layer.experts, (dispatched_input, tokens_per_expert, ctx))
            save_tensors.append(expert_output)

            output, mlp_bias = moe_layer.token_dispatcher.token_unpermutation(expert_output, mlp_bias, save_tensors)

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
                    moe_layer.router.l_expert_aux / args.moe_aux_loss_coeff,
                    moe_layer.layer_number,
                    moe_layer.config.num_layers,
                )
                if hasattr(moe_layer.router, 'l_device_aux'):
                    save_to_aux_losses_tracker(
                        "load_balancing_device_level_loss",
                        moe_layer.router.l_device_aux / args.moe_device_level_aux_loss_coeff,
                        moe_layer.layer_number,
                        moe_layer.config.num_layers,
                    )
                if hasattr(moe_layer.router, 'l_comm_aux'):
                    save_to_aux_losses_tracker(
                        "load_balancing_comm_level_loss",
                        moe_layer.router.l_comm_aux / args.moe_comm_aux_loss_coeff,
                        moe_layer.layer_number,
                        moe_layer.config.num_layers,
                    )
                output = MoEAuxLossAutoScaler.apply(output, moe_layer.router.l_aux)
        else:
            save_tensors.append(None)
            save_tensors.append(None)

        save_tensors.append(hidden_states)

        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            ctx.tokens_per_expert = tokens_per_expert

        ctx.output_splits = moe_layer.token_dispatcher.output_splits
        ctx.input_splits = moe_layer.token_dispatcher.input_splits
        ctx.router_topk = moe_layer.token_dispatcher.router_topk
        ctx.input_splits_tp_ep = getattr(moe_layer.token_dispatcher, 'input_splits_tp_ep', None)
        if n_shared_experts:
            if rs_shared_experts_handle is not None:
                rs_shared_experts_handle.wait()
            output_sum = output + rs_share_experts_output
            output.untyped_storage().resize_(0)
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
        global_args = get_args()

        output_splits = ctx.output_splits
        input_splits = ctx.input_splits
        router_topk = ctx.router_topk
        n_shared_experts = ctx.n_shared_experts
        moe_zero_memory = ctx.moe_zero_memory
        moe_experts_pipeline_degree = ctx.moe_experts_pipeline_degree
        moe_tp_extend_ep = global_args.moe_tp_extend_ep
        moe_hierarchical_alltoallv = global_args.moe_hierarchical_alltoallv
        shared_expert_gate = ctx.shared_expert_gate
        input_splits_tp_ep = ctx.input_splits_tp_ep

        (route_graph, detach_scores,
         indices, indices_ep,
         hidden_states_ep, scores_ep,
         permute1_graph,
         permute2_input_detach, permute2_graph,
         experts_graph,
         unpermute1_input_detach, unpermute1_graph,
         unpermute2_input_detach, unpermute2_graph, l_aux_graph, l_aux_detach,
         detach_input, share_experts_graph,
         global_input_tokens_local_experts_indices,
         ) = ctx.saved_tensors
        if moe_hierarchical_alltoallv:
            set_gemm_backward_need_tensors(
                ((hidden_states_ep, indices_ep, scores_ep, router_topk, global_input_tokens_local_experts_indices),
                 permute2_input_detach, permute2_graph,
                 output_splits, input_splits, input_splits_tp_ep))
        elif moe_experts_pipeline_degree:
            input_list = ctx.input_list
        else:
            set_gemm_backward_need_tensors(
                ((detach_input, indices, scores_ep, router_topk, global_input_tokens_local_experts_indices),
                 permute2_input_detach, permute2_graph,
                 output_splits, input_splits, input_splits_tp_ep))

        if n_shared_experts:
            if get_tensor_model_parallel_world_size() > 1 and not shared_expert_gate:
                _, backward_ag_shared, backward_ag_shared_handle = async_all_gather(
                    args[0], get_tensor_model_parallel_group()
                )
            else:
                backward_ag_shared = args[0]
                backward_ag_shared_handle = None

        if moe_hierarchical_alltoallv:
            ep_group = parallel_state.get_expert_model_parallel_group()
            unpermute2_graph_backward_input = args[0].view(-1, args[0].shape[-1])
            _, unpermute2_graph_backward_input, output_backward_handle = \
                async_all_gather(unpermute2_graph_backward_input, group=ep_group)
            if moe_zero_memory == "level0":
                def alltoall_token_permutation1(hidden_states, indices, router_topk):
                    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                    permutated_local_input_tokens, _, _ = permute_with_ep(
                        hidden_states, indices, probs=scores_ep, topk=router_topk, gb_inputs_splits=input_splits_tp_ep
                    )
                    return permutated_local_input_tokens

                permutated_local_input_tokens = alltoall_token_permutation1(hidden_states_ep, indices_ep, router_topk)
                set_permute_with_ep_local_input_tokens(permutated_local_input_tokens)

        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            with torch.no_grad():
                if get_tensor_model_parallel_world_size() > 1 and n_shared_experts:
                    _, shared_experts_input, shared_experts_allgather_handle = async_all_gather(
                        detach_input, get_tensor_model_parallel_group(), is_use_get_global_memory_buffer=True
                    )
                    AG_SHARED_EXPERTS_INPUTS.append((shared_experts_input, shared_experts_allgather_handle))

                # Recompute token rearrange in permutation1
                if moe_hierarchical_alltoallv:
                    permutated_local_input_tokens, _, _ = permute_with_ep(
                        hidden_states_ep.view(-1, hidden_states_ep.shape[-1]), indices_ep, probs=scores_ep, topk=ctx.router_topk,
                        gb_inputs_splits=ctx.input_splits_tp_ep
                    )
                else:
                    permutated_local_input_tokens, _ = permute(
                        detach_input.view(-1, detach_input.shape[-1]), indices
                    )

                # Recompute expert parallel AlltoAll communication
                ep_group = parallel_state.get_expert_model_parallel_group()
                if moe_tp_extend_ep:
                    ep_group = parallel_state.get_tensor_and_expert_parallel_group()
                if moe_hierarchical_alltoallv:
                    tp_group = parallel_state.get_tensor_model_parallel_group()
                    _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
                        permutated_local_input_tokens,
                        ctx.output_splits,
                        ctx.input_splits,
                        tp_group,
                    )
                else:
                    _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
                        permutated_local_input_tokens,
                        ctx.output_splits,
                        ctx.input_splits,
                        ep_group,
                    )
        if moe_hierarchical_alltoallv:
            output_backward_handle.wait()
            unpermute2_graph.backward(unpermute2_graph_backward_input)
        else:
            unpermute2_graph.backward(args[0])
        unpermute2_graph = None
        if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
            if n_shared_experts:
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
        if moe_tp_extend_ep:
            ep_group = parallel_state.get_tensor_and_expert_parallel_group()
        if moe_hierarchical_alltoallv:
            tp_group = parallel_state.get_tensor_model_parallel_group()
            _, unpermute1_backward_input, handle = async_all_to_all(
                unpermute2_input_detach.grad,
                output_splits,
                input_splits,
                tp_group,
            )
        else:
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
                    if not moe_tp_extend_ep and get_tensor_model_parallel_world_size() > 1 and ctx.moe_grouped_gemm:
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
            if moe_hierarchical_alltoallv:
                ctx.input_splits_tp_ep = None
        elif share_experts_graph is not None:
            if backward_ag_shared_handle is not None:
                backward_ag_shared_handle.wait()
            share_experts_graph.backward(backward_ag_shared)
            share_experts_graph = None
            if backward_ag_shared_handle is not None:
                backward_ag_shared.untyped_storage().resize_(0)
        if handle is not None:
            handle.wait()
            unpermute2_input_detach.grad.untyped_storage().resize_(0)

        backward_func(unpermute1_graph, unpermute1_backward_input)

        unpermute1_backward_input.untyped_storage().resize_(0)
        if moe_hierarchical_alltoallv:
            set_all2all_experts_output((permute1_graph, scores_ep, hidden_states_ep))
            backward_func(experts_graph, unpermute1_input_detach.grad)
            unpermute1_input_detach.grad.untyped_storage().resize_(0)
            permute2_input_detach.grad.untyped_storage().resize_(0)
            detach_scores_grad, detach_input_grad, detach_input_handle = get_all2all_experts_output()
        elif moe_experts_pipeline_degree:
            expert_grad_output = moe_experts_pipeline_backward_func(ctx, ctx.input_list)
            for input_tensor in input_list:
                input_tensor.untyped_storage().resize_(0)
            permute2_graph.backward(expert_grad_output)
            backward_func(permute1_graph, permute2_input_detach.grad)
            permute2_input_detach.grad.untyped_storage().resize_(0)
        else:
            backward_func(experts_graph, unpermute1_input_detach.grad)
            unpermute1_input_detach.grad.untyped_storage().resize_(0)
            permute1_backward_input, bw_permute1_ep_all2all_handle = get_all2all_experts_output()
            bw_permute1_ep_all2all_handle.wait()
            permute2_input_detach.grad.untyped_storage().resize_(0)
            backward_func(permute1_graph, permute1_backward_input)
            permute1_backward_input.untyped_storage().resize_(0)
        if l_aux_graph is not None:
            l_aux_graph.backward(l_aux_detach.grad, retain_graph=True)
        if moe_zero_memory != "disable":
            if ctx.router_topk > 1:
                stream, matmul_output_grad, unpermuted_tokens = get_prob_backward_need_tensors()
                torch.npu.current_stream().wait_stream(stream)
                probs_grad = (matmul_output_grad * unpermuted_tokens).sum(-1).squeeze(-1)
                if moe_hierarchical_alltoallv:
                    ep_group = parallel_state.get_expert_model_parallel_group()
                    _, probs_grad, probs_grad_handle = async_reduce_scatter(probs_grad, group=ep_group)
                    probs_grad_handle.wait()
                    route_graph.backward(probs_grad)
                    detach_input_handle.wait()
                else:
                    route_graph.backward(probs_grad)
            ctx.router_topk = None
        else:
            if moe_hierarchical_alltoallv:
                route_graph.backward(detach_scores_grad)
                detach_input_handle.wait()
            else:
                route_graph.backward(detach_scores.grad)
        route_graph = None
        if moe_hierarchical_alltoallv:
            grad_output = detach_input.grad + detach_input_grad
        else:
            grad_output = detach_input.grad
        return grad_output, None
