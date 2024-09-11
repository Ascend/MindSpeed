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
from megatron.core.parallel_state import get_expert_model_parallel_group, get_tensor_and_expert_parallel_group
from megatron.training import get_args
from mindspeed.ops.gmm import GMMFunction
from mindspeed.model.transformer import should_recompute_activation
from mindspeed.core.transformer.moe.moe_utils import (get_gemm_backward_need_tensors, get_ag_tp_hidden_status,
                                                      set_rs_global_hidden_states_grad_with_handle)
from mindspeed.core.transformer.moe.moe_utils import forward_func, backward_func
from mindspeed.core.transformer.moe.comm_utils import async_all_gather, async_reduce_scatter


class GroupedMlpWithCompAndCommOverlapAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights1, weights2, args):
        activation_func, group_list, layer_number = args
        if isinstance(group_list, (torch.Tensor, type(None))):
            group_list_data_type = 1
        else:
            group_list_data_type = 0
        ctx.group_list_data_type = group_list_data_type
        mm1_out = \
            GMMFunction.builder.load().npu_gmm([inputs], [weights1], [], group_list.tolist(), 0, group_list_data_type)[
                0]
        inputs.untyped_storage().resize_(0)
        act_out, detached_act_inputs = forward_func(activation_func, mm1_out)
        mm2_out = \
            GMMFunction.builder.load().npu_gmm([act_out], [weights2], [], group_list.tolist(), 0, group_list_data_type)[
                0]
        if should_recompute_activation(layer_number):
            act_out.untyped_storage().resize_(0)
            ctx.activation_func = activation_func
        ctx.layer_number = layer_number
        ctx.save_for_backward(detached_act_inputs, act_out, weights1, weights2, group_list)
        return mm2_out, None

    @staticmethod
    def backward(ctx, *grad_outs):
        grad_outs = grad_outs[0]
        layer_number = ctx.layer_number
        act_inputs, act_graph, weights1, weights2, group_list = ctx.saved_tensors
        group_list_data_type = ctx.group_list_data_type
        token_unpermutation_graph, global_hidden_states_detach, indices, global_local_map = get_gemm_backward_need_tensors()

        # grad of mm2
        weights2 = rearrange(weights2, 'n h f -> n f h')

        grad_mm2_inputs = \
            GMMFunction.builder.load().npu_gmm([grad_outs], [weights2], [], group_list.tolist(), 0,
                                               group_list_data_type)[0]
        if should_recompute_activation(layer_number):
            activation_func = ctx.activation_func
            act_out = activation_func(act_inputs)
            mm2_inputs = act_out
        else:
            mm2_inputs = act_graph
        grad_weights2 = GMMFunction.builder.load().npu_gmm([mm2_inputs.t()], [grad_outs], [], group_list.tolist(), 2,
                                                           group_list_data_type)[0]
        grad_outs.untyped_storage().resize_(0)
        mm2_inputs.untyped_storage().resize_(0)

        # grad of activation_func
        act_graph.backward(grad_mm2_inputs)
        grad_mm2_inputs.untyped_storage().resize_(0)
        act_inputs.untyped_storage().resize_(0)
        mm1_outs_grad = act_inputs.grad

        # re-gather mm1 forward inputs
        ag_inputs_tp = get_ag_tp_hidden_status()
        ag_inputs_tp = ag_inputs_tp.view(-1, ag_inputs_tp.shape[-1])
        # mm1_inputs = ctx.inputs
        _, ag_inputs_tp_ep, ag_handle = async_all_gather(ag_inputs_tp,
                                                         get_expert_model_parallel_group() if get_args().n_shared_experts else get_tensor_and_expert_parallel_group())

        # grad of mm1-inputs
        weights1 = rearrange(weights1, 'n h f -> n f h')
        mm1_inputs_grad = GMMFunction.builder.load().npu_gmm([mm1_outs_grad], [weights1], [], group_list.tolist(), 0,
                                                             group_list_data_type)[0]
        # token 反重排的反向

        backward_func(token_unpermutation_graph, mm1_inputs_grad)
        mm1_inputs_grad.untyped_storage().resize_(0)
        _, rs_global_hidden_states_grad, rs_handle = async_reduce_scatter(global_hidden_states_detach.grad,
                                                                          get_tensor_and_expert_parallel_group())
        rs_global_hidden_states_grad_with_handle = (rs_global_hidden_states_grad, rs_handle)
        ag_handle.wait()
        # token 重排计算
        mm1_inputs = ag_inputs_tp_ep[global_local_map, :][indices, :]
        global_local_map.untyped_storage().resize_(0)
        indices.untyped_storage().resize_(0)
        ag_inputs_tp_ep.untyped_storage().resize_(0)
        mm1_weights_grad = \
            GMMFunction.builder.load().npu_gmm([mm1_inputs.t()], [mm1_outs_grad], [], group_list.tolist(), 2,
                                               group_list_data_type)[0]
        mm1_outs_grad.untyped_storage().resize_(0)

        set_rs_global_hidden_states_grad_with_handle(rs_global_hidden_states_grad_with_handle)
        return mm1_inputs_grad, mm1_weights_grad, grad_weights2, None


def grouped_mlp_with_comp_and_comm_overlap_allgather(inputs, weights1, weights2, args):
    return GroupedMlpWithCompAndCommOverlapAllGather.apply(inputs, weights1, weights2, args)
