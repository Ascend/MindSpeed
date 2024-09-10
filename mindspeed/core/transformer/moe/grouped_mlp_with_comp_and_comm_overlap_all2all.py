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
from megatron.core.parallel_state import get_expert_model_parallel_group
from megatron.core.transformer.moe.moe_utils import permute
from mindspeed.ops.gmm import GMMFunction
from mindspeed.core.transformer.moe.moe_layer_overlap_all2all import forward_func, backward_func
from mindspeed.core.transformer.moe.comm_utils import async_all_to_all


class GroupedMlpWithCompAndCommOverlapAll2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights1, weights2, activation_func, group_list, moe_layer_ctx):
        if isinstance(group_list, (torch.Tensor, type(None))):
            group_list_data_type = 1
        else:
            group_list_data_type = 0
        ctx.group_list_data_type = group_list_data_type
        mm1_out = GMMFunction.builder.load().npu_gmm([inputs], [weights1], [], group_list.tolist(), 0, group_list_data_type)[0]
        inputs.untyped_storage().resize_(0)
        act_out, detached_act_inputs = forward_func(activation_func, mm1_out)
        args = get_args()
        moe_without_activation = args.moe_without_activation
        if moe_without_activation:
            mm1_out.untyped_storage().resize_(0)
        mm2_out = GMMFunction.builder.load().npu_gmm([act_out], [weights2], [], group_list.tolist(), 0, group_list_data_type)[0]
        if moe_without_activation:
            act_out.untyped_storage().resize_(0)
            moe_layer_ctx.recompute_tensors = (inputs, mm1_out, act_out)
        if moe_without_activation:
            ctx.save_for_backward(inputs, detached_act_inputs, act_out, weights1, weights2, group_list)
        else:
            ctx.detached_act_inputs = detached_act_inputs
            ctx.save_for_backward(detached_act_inputs, act_out, weights1, weights2, group_list)

        return mm2_out, None

    @staticmethod
    def backward(ctx, *grad_outs):
        grad_outs = grad_outs[0]
        args = get_args()
        moe_without_activation = args.moe_without_activation
        if moe_without_activation:
            inputs, act_inputs, mm2_inputs, weights1, weights2, group_list = ctx.saved_tensors
        else:
            _, mm2_inputs, weights1, weights2, group_list = ctx.saved_tensors
            act_inputs = ctx.detached_act_inputs
        group_list_data_type = ctx.group_list_data_type
        from mindspeed.core.transformer.moe.moe_utils import get_gemm_backward_need_tensors, set_all2all_experts_output
        ((detach_input, indices, router_topk, global_input_tokens_local_experts_indices),
         permute2_input_detach, permute2_graph, output_splits, input_splits) = get_gemm_backward_need_tensors()

        # grad of mm2
        weights2 = rearrange(weights2, 'n h f -> n f h')
        grad_mm2_inputs = GMMFunction.builder.load().npu_gmm([grad_outs], [weights2], [], group_list.tolist(), 0, group_list_data_type)[0]
        grad_weights2 = GMMFunction.builder.load().npu_gmm([mm2_inputs.t()], [grad_outs], [], group_list.tolist(), 2, group_list_data_type)[0]
        # grad of activation_func
        grad_outs.untyped_storage().resize_(0)
        mm2_inputs.untyped_storage().resize_(0)
        mm2_inputs.backward(grad_mm2_inputs)
        grad_mm2_inputs.untyped_storage().resize_(0)
        if not moe_without_activation:

            def alltoall_token_permutation1(hidden_states, indices, router_topk):
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                permutated_local_input_tokens, _ = permute(
                    hidden_states, indices, topk=router_topk
                )
                return permutated_local_input_tokens

            permutated_local_input_tokens = alltoall_token_permutation1(detach_input, indices, router_topk)
        detach_input.untyped_storage().resize_(0)
        if not moe_without_activation:
            _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
                permutated_local_input_tokens,
                output_splits,
                input_splits,
                get_expert_model_parallel_group(),
            )
        weights1 = rearrange(weights1, 'n h f -> n f h')
        mm1_inputs_grad = GMMFunction.builder.load().npu_gmm([act_inputs.grad], [weights1], [], group_list.tolist(), 0, group_list_data_type)[0]

        # 峰值
        backward_func(permute2_graph, mm1_inputs_grad)
        _, permute1_backward_input, bw_permute1_ep_all2all_handle = async_all_to_all(
            permute2_input_detach.grad,
            input_splits,
            output_splits,
            get_expert_model_parallel_group(),
        )

        set_all2all_experts_output((permute1_backward_input, bw_permute1_ep_all2all_handle))
        if not moe_without_activation:
            permute1_ep_all_to_all_handle.wait()

            permutated_local_input_tokens.untyped_storage().resize_(0)

            mm1_inputs, _ = permute(
                global_input_tokens, global_input_tokens_local_experts_indices
            )

            global_input_tokens.untyped_storage().resize_(0)

        if moe_without_activation:
            mm1_weights_grad = GMMFunction.builder.load().npu_gmm([inputs.t()], [act_inputs.grad], [], group_list.tolist(), 2, group_list_data_type)[0]
        else:
            mm1_weights_grad = GMMFunction.builder.load().npu_gmm([mm1_inputs.t()], [act_inputs.grad], [], group_list.tolist(), 2, group_list_data_type)[0]
        act_inputs.grad.untyped_storage().resize_(0)
        return mm1_inputs_grad, mm1_weights_grad, grad_weights2, None, None, None


def grouped_mlp_with_comp_and_comm_overlap_all2all(inputs, weights1, weights2, activation_func, group_list, ctx):
    return GroupedMlpWithCompAndCommOverlapAll2All.apply(inputs, weights1, weights2, activation_func, group_list, ctx)
