# Copyright (c) 2025, Huawei Technologies. All rights reserved.

import torch
import acl
from einops import rearrange

from mindspeed.core.transformer.moe.grouped_matmul_util import get_gmm_op_cls
from mindspeed.core.transformer.moe.moe_feature import (
    permute,
    parallel_state
)
from mindspeed.ops.gmm import GMMFunction
from mindspeed.model.transformer import should_recompute_activation
from mindspeed.core.transformer.moe.moe_feature.overlap.moe_common import (
    forward_func, backward_func,
    get_gemm_backward_need_tensors, get_ag_tp_hidden_status,
    set_rs_global_hidden_states_grad_with_handle,
    )
from mindspeed.core.transformer.moe.moe_feature.overlap.comm_utils import async_all_gather, async_reduce_scatter
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32


class GroupedMlpWithCompAndCommOverlapAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights1, weights2, args):
        original_weight1, original_weight2, activation_func, group_list, layer_number, config = args
        ctx.config = config        
        use_gmm = (inputs.nelement() != 0)
        ctx.use_gmm = use_gmm
        if isinstance(group_list, torch.Tensor):
            if group_list.device.type == 'cpu':
                group_list = group_list.npu()
        gmm_cls = get_gmm_op_cls()
        if use_gmm:
            mm1_out = gmm_cls.op_forward(inputs, weights1, group_list)[0]
        else:
            mm1_out = torch.matmul(inputs, weights1)
        inputs.untyped_storage().resize_(0)
        act_out, detached_act_inputs = forward_func(activation_func, mm1_out)
        if use_gmm:
            mm2_out = gmm_cls.op_forward(act_out, weights2, group_list)[0]
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
        config = ctx.config
        act_inputs, act_graph, weights1, weights2, original_weight1, original_weight2, group_list = ctx.saved_tensors
        token_unpermutation_graph, global_hidden_states_detach, local_map, reversed_local_input_permutation_mapping = get_gemm_backward_need_tensors()
        gmm_cls = get_gmm_op_cls()
        # grad of mm2
        if ctx.use_gmm:
            grad_mm2_inputs = gmm_cls.op_dx(grad_outs, weights2, group_list)[0]
        else:
            grad_mm2_inputs = torch.matmul(grad_outs, weights2.t())
        if should_recompute_activation(layer_number):
            activation_func = ctx.activation_func
            act_out = activation_func(act_inputs)
            mm2_inputs = act_out
        else:
            mm2_inputs = act_graph
        
        if ctx.use_gmm:
            if config.gemm_gradient_accumulation_fusion:
                grad_weights2 = gmm_cls.op_gmm_add(mm2_inputs, weights2, grad_outs, group_list, original_weight2)
            else:
                grad_weights2 = gmm_cls.op_dw(mm2_inputs, grad_outs, group_list)[0]
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
        ag_group = parallel_state.get_expert_tensor_and_model_parallel_group()
        _, ag_inputs_tp_ep, ag_handle = async_all_gather(ag_inputs_tp, ag_group)
        if ctx.use_gmm:
            # grad of mm1-inputs
            mm1_inputs_grad = gmm_cls.op_dx(act_inputs.grad, weights1, group_list)[0]
        else:
            mm1_inputs_grad = torch.matmul(act_inputs.grad, weights1.t())

        # token unpermute backward.
        backward_func(token_unpermutation_graph, mm1_inputs_grad)
        mm1_inputs_grad.untyped_storage().resize_(0)
        _, rs_global_hidden_states_grad, rs_handle = async_reduce_scatter(global_hidden_states_detach.grad,
                                                                          parallel_state.get_expert_tensor_and_model_parallel_group())
        rs_global_hidden_states_grad_with_handle = (rs_global_hidden_states_grad, rs_handle)
        ag_handle.wait()

        # token re-premute.

        (mm1_inputs, _, _) = permute(
            ag_inputs_tp_ep, local_map
        )

        local_map.untyped_storage().resize_(0)
        ag_inputs_tp_ep.untyped_storage().resize_(0)

        if ctx.use_gmm:
            if config.gemm_gradient_accumulation_fusion:
                mm1_weights_grad = gmm_cls.op_gmm_add(mm1_inputs, weights1, act_inputs.grad, group_list,
                                                      original_weight1)
            else:
                mm1_weights_grad = gmm_cls.op_dw(mm1_inputs, act_inputs.grad, group_list)
        else:
            mm1_weights_grad = torch.matmul(mm1_inputs.t(), act_inputs.grad)

        mm1_outs_grad.untyped_storage().resize_(0)

        set_rs_global_hidden_states_grad_with_handle(rs_global_hidden_states_grad_with_handle)
        return mm1_inputs_grad, mm1_weights_grad, grad_weights2, None


def grouped_mlp_with_comp_and_comm_overlap_allgather(inputs, weights1, weights2, args):
    return GroupedMlpWithCompAndCommOverlapAllGather.apply(inputs, weights1, weights2, args)
