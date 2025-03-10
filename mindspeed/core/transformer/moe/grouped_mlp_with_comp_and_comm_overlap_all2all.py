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
from einops import rearrange
from megatron.training import get_args
from megatron.core import parallel_state
from megatron.core.parallel_state import get_expert_model_parallel_group, get_tensor_and_expert_parallel_group, get_tensor_model_parallel_group
from megatron.core.transformer.moe.moe_utils import permute
from mindspeed.model.transformer import should_recompute_activation
from mindspeed.core.transformer.moe.moe_layer_overlap_all2all import gmm_op
from mindspeed.core.transformer.moe.comm_utils import (async_all_to_all, async_reduce_scatter, async_all_gather,
                                                       transfer_tensor_last_dim_to_first)
from mindspeed.core.transformer.moe.moe_utils import (only_recompute_activation, forward_func, backward_func,
                                                      get_gemm_backward_need_tensors, 
                                                      set_all2all_experts_output, 
                                                      permute_with_ep, get_all2all_experts_output,
                                                      get_permute_with_ep_local_input_tokens)
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32


class GroupedMlpWithCompAndCommOverlapAll2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights1, weights2, args, moe_layer_ctx):
        original_weight1, original_weight2, activation_func, group_list, layer_number = args
        global_args = get_args()
        moe_zero_memory = global_args.moe_zero_memory
        moe_experts_pipeline_degree = global_args.moe_experts_pipeline_degree
        ctx.layer_number = layer_number
        ctx.moe_zero_memory = moe_zero_memory
        ctx.moe_experts_pipeline_degree = moe_experts_pipeline_degree
        use_gmm = (inputs.nelement() != 0)
        ctx.use_gmm = use_gmm
        if use_gmm:
            mm1_out = gmm_op(inputs, weights1, [], group_list, 0)[0]
        else:
            mm1_out = torch.matmul(inputs, weights1)
        if moe_zero_memory != "disable" or moe_experts_pipeline_degree:
            inputs.untyped_storage().resize_(0)
        act_out, detached_act_inputs = forward_func(activation_func, mm1_out)

        is_only_recompute_activation = only_recompute_activation(layer_number)
        if moe_zero_memory == "level1" and not is_only_recompute_activation:
            mm1_out.untyped_storage().resize_(0)
        if use_gmm:
            mm2_out = gmm_op(act_out, weights2, [], group_list, 0)[0]
        else:
            mm2_out = torch.matmul(act_out, weights2)

        if moe_zero_memory == "level1" and not is_only_recompute_activation:
            act_out.untyped_storage().resize_(0)
            moe_layer_ctx.recompute_tensors = (inputs, mm1_out, act_out)
        is_recompute_activation = moe_zero_memory == "level0" or should_recompute_activation(layer_number) or (
                    moe_zero_memory == "level1" and is_only_recompute_activation)
        if is_recompute_activation:
            act_out.untyped_storage().resize_(0)
            ctx.activation_func = activation_func
        if moe_zero_memory != "level0" and not (moe_zero_memory == "level1" and is_only_recompute_activation):
            ctx.save_for_backward(inputs, detached_act_inputs, act_out, weights1, weights2, original_weight1,
                                  original_weight2, group_list)
        else:
            ctx.save_for_backward(detached_act_inputs, act_out, weights1, weights2, original_weight1, original_weight2,
                                  group_list)

        return mm2_out, None

    @staticmethod
    def backward(ctx, *grad_outs):
        grad_outs = grad_outs[0]
        global_args = get_args()
        moe_hierarchical_alltoallv = global_args.moe_hierarchical_alltoallv
        layer_number = ctx.layer_number
        moe_zero_memory = ctx.moe_zero_memory
        moe_experts_pipeline_degree = ctx.moe_experts_pipeline_degree
        is_only_recompute_activation = only_recompute_activation(layer_number)
        if moe_zero_memory != "level0" and not (moe_zero_memory == "level1" and is_only_recompute_activation):
            mm1_inputs, act_inputs, mm2_inputs, weights1, weights2, original_weight1, original_weight2, group_list = ctx.saved_tensors
        else:
            act_inputs, mm2_inputs, weights1, weights2, original_weight1, original_weight2, group_list = ctx.saved_tensors
        if moe_experts_pipeline_degree:
            inputs_save = get_gemm_backward_need_tensors()
            _, inputs, ag_handle_i = async_all_gather(inputs_save, get_tensor_model_parallel_group(()), last_dim=True)
        else:
            ((detach_input, indices, scores_ep, router_topk, global_input_tokens_local_experts_indices),
             permute2_input_detach, permute2_graph, output_splits, input_splits,
             input_splits_tp_ep) = get_gemm_backward_need_tensors()

        # grad of mm2 dx
        if ctx.use_gmm:
            weights2 = rearrange(weights2, 'n h f -> n f h')
            grad_mm2_inputs = gmm_op(grad_outs, weights2, [], group_list, 0)[0]
        else:
            grad_mm2_inputs = torch.matmul(grad_outs, weights2.t())
        act_graph = mm2_inputs
        is_recompute_activation = moe_zero_memory == "level0" or should_recompute_activation(layer_number) or (
                    moe_zero_memory == "level1" and is_only_recompute_activation)
        if is_recompute_activation:
            activation_func = ctx.activation_func
            mm2_inputs = activation_func(act_inputs)

        if moe_hierarchical_alltoallv:
            ep_group = parallel_state.get_expert_model_parallel_group()
            tp_group = parallel_state.get_tensor_model_parallel_group()
            permute1_graph, scores_ep, hidden_states_ep = get_all2all_experts_output()
            if moe_zero_memory == "disable":
                _, detach_scores_grad, detach_scores_handle = async_reduce_scatter(scores_ep.grad, group=ep_group)
            else:
                detach_scores_grad = None
                detach_scores_handle = None

            # grad of activation_func
            act_graph.backward(grad_mm2_inputs)
            if moe_zero_memory == "level0" or (moe_zero_memory == "level1" and is_only_recompute_activation):
                permutated_local_input_tokens = get_permute_with_ep_local_input_tokens()
                _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
                    permutated_local_input_tokens,
                    output_splits,
                    input_splits,
                    tp_group,
                )

            # gmm1 dx
            if ctx.use_gmm:
                weights1 = rearrange(weights1, 'n h f -> n f h')
                mm1_inputs_grad = \
                    gmm_op(act_inputs.grad, weights1, [], group_list, 0)[0]
            else:
                mm1_inputs_grad = torch.matmul(act_inputs.grad, weights1.t())

            backward_func(permute2_graph, mm1_inputs_grad)
            mm1_inputs_grad.untyped_storage().resize_(0)

            if moe_zero_memory == "level0" or (moe_zero_memory == "level1" and is_only_recompute_activation):
                permute1_ep_all_to_all_handle.wait()
                permutated_local_input_tokens.untyped_storage().resize_(0)
            _, permute1_backward_input, bw_permute1_ep_all2all_handle = async_all_to_all(
                permute2_input_detach.grad,
                input_splits,
                output_splits,
                tp_group,
            )

        # gmm2 dw
        if ctx.use_gmm:
            if get_args().gemm_gradient_accumulation_fusion:

                npu_groupmatmul_add_fp32(mm2_inputs, grad_outs, group_list, original_weight2.main_grad)

                if hasattr(original_weight2, 'grad_added_to_main_grad'):
                    if getattr(weights2, 'zero_out_wgrad', False):
                        grad_weights2 = torch.zeros(
                            weights2.transpose(-1, -2).shape,
                            dtype=mm2_inputs.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                    else:
                        grad_weights2 = torch.empty(
                            weights2.transpose(-1, -2).shape,
                            dtype=mm2_inputs.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                    original_weight2.grad_added_to_main_grad = True
                else:
                    grad_weights2 = None
            else:
                grad_weights2 = gmm_op(mm2_inputs.t(), grad_outs, [], group_list, 2)[0]
        else:
            grad_weights2 = torch.matmul(mm2_inputs.t(), grad_outs)

        # grad of activation_func
        grad_outs.untyped_storage().resize_(0)
        mm2_inputs.untyped_storage().resize_(0)
        if moe_hierarchical_alltoallv:
            grad_mm2_inputs.untyped_storage().resize_(0)
            act_inputs.untyped_storage().resize_(0)
            bw_permute1_ep_all2all_handle.wait()

            backward_func(permute1_graph, permute1_backward_input)
            permute1_backward_input.untyped_storage().resize_(0)
            if moe_zero_memory == "disable":
                detach_scores_handle.wait()

            ep_group = parallel_state.get_expert_model_parallel_group()
            _, detach_input_grad, detach_input_handle = async_reduce_scatter(hidden_states_ep.grad, group=ep_group)
            set_all2all_experts_output((detach_scores_grad, detach_input_grad, detach_input_handle))
        else:
            act_graph.backward(grad_mm2_inputs)
            grad_mm2_inputs.untyped_storage().resize_(0)
            act_inputs.untyped_storage().resize_(0)
            if moe_zero_memory == "level0" or (moe_zero_memory == "level1" and is_only_recompute_activation):
                def alltoall_token_permutation1(hidden_states, indices):
                    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                    permutated_local_input_tokens, _ = permute(
                        hidden_states, indices
                    )
                    return permutated_local_input_tokens

                permutated_local_input_tokens = alltoall_token_permutation1(detach_input, indices)

                ep_group = get_expert_model_parallel_group()
                if global_args.moe_tp_extend_ep:
                    ep_group = get_tensor_and_expert_parallel_group()
                _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
                    permutated_local_input_tokens,
                    output_splits,
                    input_splits,
                    ep_group,
                )
            if ctx.use_gmm:
                weights1 = rearrange(weights1, 'n h f -> n f h')
                mm1_inputs_grad = gmm_op(act_inputs.grad, weights1, [], group_list, 0)[0]
            else:
                mm1_inputs_grad = torch.matmul(act_inputs.grad, weights1.t())

            # 峰值
            if moe_experts_pipeline_degree:
                ag_handle_i.wait()
                mm1_inputs = torch.cat(inputs, dim=inputs_save.dim() - 1).contiguous()
            else:
                backward_func(permute2_graph, mm1_inputs_grad)
                mm1_inputs_grad.untyped_storage().resize_(0)
                ep_group = get_expert_model_parallel_group()
                if global_args.moe_tp_extend_ep:
                    ep_group = get_tensor_and_expert_parallel_group()

            if moe_zero_memory == "level0" or (moe_zero_memory == "level1" and is_only_recompute_activation):
                permute1_ep_all_to_all_handle.wait()
                permutated_local_input_tokens.untyped_storage().resize_(0)

            if moe_experts_pipeline_degree:
                mm1_inputs_grad, num_dim = transfer_tensor_last_dim_to_first(mm1_inputs_grad)
                rs_input_i, expert_output, rs_handle_i = async_reduce_scatter(mm1_inputs_grad,
                                                                              get_tensor_model_parallel_group())
                set_all2all_experts_output((rs_input_i, expert_output, rs_handle_i, mm1_inputs_grad, num_dim))
            else:
                _, permute1_backward_input, bw_permute1_ep_all2all_handle = async_all_to_all(
                    permute2_input_detach.grad,
                    input_splits,
                    output_splits,
                    ep_group,
                )
                set_all2all_experts_output((permute1_backward_input, bw_permute1_ep_all2all_handle))

        if moe_zero_memory == "level0" or (moe_zero_memory == "level1" and is_only_recompute_activation):
            mm1_inputs, _ = permute(
                global_input_tokens, global_input_tokens_local_experts_indices
            )

            global_input_tokens.untyped_storage().resize_(0)

        if ctx.use_gmm:
            if get_args().gemm_gradient_accumulation_fusion:
                npu_groupmatmul_add_fp32(mm1_inputs, act_inputs.grad, group_list, original_weight1.main_grad)
                if hasattr(original_weight1, 'grad_added_to_main_grad'):
                    if getattr(weights1, 'zero_out_wgrad', False):
                        mm1_weights_grad = torch.zeros(
                            weights1.transpose(-1, -2).shape,
                            dtype=mm1_inputs.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                    else:
                        mm1_weights_grad = torch.empty(
                            weights1.transpose(-1, -2).shape,
                            dtype=mm1_inputs.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                    original_weight1.grad_added_to_main_grad = True
                else:
                    mm1_weights_grad = None
            else:
                mm1_weights_grad = gmm_op(mm1_inputs.t(), act_inputs.grad, [], group_list, 2)[0]
        else:
            mm1_weights_grad = torch.matmul(mm1_inputs.t(), act_inputs.grad)
        act_inputs.grad.untyped_storage().resize_(0)
        if moe_experts_pipeline_degree:
            return None, mm1_weights_grad, grad_weights2, None, None
        else:
            return mm1_inputs_grad, mm1_weights_grad, grad_weights2, None, None


def grouped_mlp_with_comp_and_comm_overlap_all2all(inputs, weights1, weights2, args, ctx):
    return GroupedMlpWithCompAndCommOverlapAll2All.apply(inputs, weights1, weights2, args, ctx)
