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
import acl
from einops import rearrange
from megatron.core.parallel_state import get_expert_model_parallel_group, get_tensor_and_expert_parallel_group, get_tensor_and_expert_parallel_world_size, get_expert_model_parallel_world_size
from megatron.training import get_args
from mindspeed.ops.gmm import GMMFunction
from mindspeed.model.transformer import should_recompute_activation
from mindspeed.core.transformer.moe.moe_utils import (get_gemm_backward_need_tensors, get_ag_tp_hidden_status,
                                                      set_rs_global_hidden_states_grad_with_handle)
from mindspeed.core.transformer.moe.moe_utils import forward_func, backward_func
from mindspeed.core.transformer.moe.comm_utils import async_all_gather, async_reduce_scatter
from mindspeed.core.transformer.moe.token_dispatcher import cann_version_check
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32
from .moe_layer_overlap_all2all import gmm_op


class GroupedMlpWithCompAndCommOverlapAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights1, weights2, args):
        original_weight1, original_weight2, activation_func, group_list, layer_number = args
        use_gmm = (inputs.nelement() != 0)
        ctx.use_gmm = use_gmm
        if use_gmm:
            mm1_out = gmm_op(inputs, weights1, [], group_list, 0)[0]
        else:
            mm1_out = torch.matmul(inputs, weights1)
        inputs.untyped_storage().resize_(0)
        act_out, detached_act_inputs = forward_func(activation_func, mm1_out)
        if use_gmm:
            mm2_out = gmm_op(act_out, weights2, [], group_list, 0)[0]
        else:
            mm2_out = torch.matmul(act_out, weights2)
        if should_recompute_activation(layer_number):
            act_out.untyped_storage().resize_(0)
            ctx.activation_func = activation_func
        ctx.layer_number = layer_number
        ctx.save_for_backward(detached_act_inputs, act_out, weights1, weights2, original_weight1, original_weight2, group_list)
        return mm2_out, None

    @staticmethod
    def backward(ctx, *grad_outs):
        grad_outs = grad_outs[0]
        layer_number = ctx.layer_number
        act_inputs, act_graph, weights1, weights2, original_weight1, original_weight2, group_list = ctx.saved_tensors
        token_unpermutation_graph, global_hidden_states_detach, indices, global_local_map = get_gemm_backward_need_tensors()

        # grad of mm2
        if ctx.use_gmm:
            weights2 = rearrange(weights2, 'n h f -> n f h')
            grad_mm2_inputs = gmm_op(grad_outs, weights2, [], group_list, 0)[0]
        else:
            grad_mm2_inputs = torch.matmul(grad_outs, weights2.t())
        if should_recompute_activation(layer_number):
            activation_func = ctx.activation_func
            act_out = activation_func(act_inputs)
            mm2_inputs = act_out
        else:
            mm2_inputs = act_graph
        
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
        ag_group = get_expert_model_parallel_group()
        if '910B' in acl.get_soc_name() or not get_args().n_shared_experts:
            ag_group = get_tensor_and_expert_parallel_group()
        _, ag_inputs_tp_ep, ag_handle = async_all_gather(ag_inputs_tp, ag_group)
        if ctx.use_gmm:
            # grad of mm1-inputs
            weights1 = rearrange(weights1, 'n h f -> n f h')
            mm1_inputs_grad = gmm_op(act_inputs.grad, weights1, [], group_list, 0)[0]
        else:
            mm1_inputs_grad = torch.matmul(act_inputs.grad, weights1.t())

        # token 反重排的反向
        backward_func(token_unpermutation_graph, mm1_inputs_grad)
        mm1_inputs_grad.untyped_storage().resize_(0)
        _, rs_global_hidden_states_grad, rs_handle = async_reduce_scatter(global_hidden_states_detach.grad,
                                                                          get_tensor_and_expert_parallel_group())
        rs_global_hidden_states_grad_with_handle = (rs_global_hidden_states_grad, rs_handle)
        ag_handle.wait()

        # token 重排计算
        global_args = get_args()
        num_local_experts = global_args.num_experts // get_expert_model_parallel_world_size()
        if global_args.moe_tp_extend_ep:
            num_local_experts = global_args.num_experts // get_tensor_and_expert_parallel_world_size()
        if cann_version_check:
            mm1_inputs = ag_inputs_tp_ep[global_local_map, :]
            if num_local_experts > 1:
                mm1_inputs = mm1_inputs[indices, :]
        else:
            mm1_inputs = torch.gather(ag_inputs_tp_ep, 0, global_local_map)
            if num_local_experts > 1:
                mm1_inputs = torch.gather(mm1_inputs, 0, indices)

        global_local_map.untyped_storage().resize_(0)
        indices.untyped_storage().resize_(0)
        ag_inputs_tp_ep.untyped_storage().resize_(0)

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

        mm1_outs_grad.untyped_storage().resize_(0)

        set_rs_global_hidden_states_grad_with_handle(rs_global_hidden_states_grad_with_handle)
        return mm1_inputs_grad, mm1_weights_grad, grad_weights2, None


def grouped_mlp_with_comp_and_comm_overlap_allgather(inputs, weights1, weights2, args):
    return GroupedMlpWithCompAndCommOverlapAllGather.apply(inputs, weights1, weights2, args)
