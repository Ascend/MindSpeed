# Copyright (c) 2025, Huawei Technologies. All rights reserved.

import torch
from einops import rearrange

from mindspeed.core.transformer.moe.grouped_matmul_util import get_gmm_op_cls
from mindspeed.model.transformer import should_recompute_activation
from mindspeed.core.transformer.moe.moe_feature.overlap.comm_utils import async_all_to_all
from mindspeed.core.transformer.moe.moe_feature.overlap.moe_common import (only_recompute_activation, 
                                                                           forward_func, backward_func,
                                                                           get_gemm_backward_need_tensors, set_all2all_experts_output,
                                                                          )
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32
from mindspeed.core.transformer.moe.moe_feature import (
    permute,
    sort_chunks_by_idxs,
    parallel_state
    )


class GroupedMlpWithCompAndCommOverlapAll2AllSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights1, weights2, args, moe_layer_ctx):
        original_weight1, original_weight2, activation_func, permuted_probs, group_list, layer_number, config = args
        moe_zero_memory = config.moe_zero_memory
        ctx.config = config
        ctx.layer_number = layer_number
        ctx.moe_zero_memory = moe_zero_memory
        ctx.activation_func = activation_func
        use_gmm = (inputs.nelement() != 0)
        ctx.use_gmm = use_gmm
        gmm_cls = get_gmm_op_cls()
        if use_gmm:
            mm1_out = gmm_cls.op_forward(inputs, weights1, group_list)[0]
        else:
            mm1_out = torch.matmul(inputs, weights1)
        if moe_zero_memory != "disable":
            inputs.untyped_storage().resize_(0)

        def activation_func_with_probs_detach(x, probs):
            dtype = x.dtype
            act_without_probs = activation_func(x)
            fin_res = act_without_probs * (probs.unsqueeze(-1))
            return fin_res.to(dtype), act_without_probs

        (act_out, act_without_probs), detached_act_inputs, permuted_probs_inputs_detach = forward_func(activation_func_with_probs_detach, (mm1_out, permuted_probs))

        is_only_recompute_activation = only_recompute_activation(config, layer_number)
        if moe_zero_memory == "level1" and not is_only_recompute_activation:
            mm1_out.untyped_storage().resize_(0)
            permuted_probs.untyped_storage().resize_(0)
        if use_gmm:
            mm2_out = gmm_cls.op_forward(act_out, weights2, group_list)[0]
        else:
            mm2_out = torch.matmul(act_out, weights2)

        if moe_zero_memory == "level1" and not is_only_recompute_activation:
            act_without_probs.untyped_storage().resize_(0)
            act_out.untyped_storage().resize_(0)
            moe_layer_ctx.recompute_tensors = (inputs, mm1_out, permuted_probs, act_out, act_without_probs)

        is_recompute_activation = moe_zero_memory == "level0" or should_recompute_activation(layer_number) or (
                    moe_zero_memory == "level1" and is_only_recompute_activation)
        if is_recompute_activation:
            act_without_probs.untyped_storage().resize_(0)
            act_out.untyped_storage().resize_(0)
        if moe_zero_memory != "level0" and not (moe_zero_memory == "level1" and is_only_recompute_activation):
            ctx.save_for_backward(inputs, permuted_probs_inputs_detach, detached_act_inputs, act_out, act_without_probs, weights1, weights2, original_weight1,
                                  original_weight2, group_list)
        else:
            ctx.save_for_backward(permuted_probs_inputs_detach, detached_act_inputs, act_out, act_without_probs, weights1, weights2, original_weight1, 
                                  original_weight2, group_list)

        return mm2_out, None

    @staticmethod
    def backward(ctx, *grad_outs):
        grad_outs = grad_outs[0]
        config = ctx.config
        layer_number = ctx.layer_number
        moe_zero_memory = ctx.moe_zero_memory
        is_only_recompute_activation = only_recompute_activation(config, layer_number)
        if moe_zero_memory != "level0" and not (moe_zero_memory == "level1" and is_only_recompute_activation):
            mm1_inputs, permuted_probs_inputs_detach, act_inputs, mm2_inputs, act_without_probs, weights1, weights2, original_weight1, original_weight2, group_list = ctx.saved_tensors
        else:
            permuted_probs_inputs_detach, act_inputs, mm2_inputs, act_without_probs, weights1, weights2, original_weight1, original_weight2, group_list = ctx.saved_tensors

        ((detach_input, probs, routing_map, num_global_tokens_per_local_expert_cpu, sort_input_by_local_experts),
         permute2_input_detach, permute2_graph, 
         permute2_prob_detach, permute2_prob_graph,
         output_splits, input_splits, num_out_tokens) = get_gemm_backward_need_tensors()

        ep_group = parallel_state.get_expert_model_parallel_group()
        if config.moe_tp_extend_ep:
            ep_group = parallel_state.get_expert_tensor_and_model_parallel_group()
        gmm_cls = get_gmm_op_cls()
        # grad of mm2
        if ctx.use_gmm:
            grad_mm2_inputs = gmm_cls.op_dx(grad_outs, weights2, group_list)[0]
        else:
            grad_mm2_inputs = torch.matmul(grad_outs, weights2.t())
        # act_graph is act_out.
        act_graph = mm2_inputs
        is_recompute_activation = moe_zero_memory == "level0" or should_recompute_activation(layer_number) or (
                    moe_zero_memory == "level1" and is_only_recompute_activation)
        if is_recompute_activation:
            dtype = act_inputs.dtype
            activation_func = ctx.activation_func
            act_without_probs_ = activation_func(act_inputs)
            mm2_inputs = act_without_probs_ * permuted_probs_inputs_detach.unsqueeze(-1)
            mm2_inputs = mm2_inputs.to(dtype)
            act_without_probs_size = act_without_probs_.untyped_storage().size()
            act_without_probs.untyped_storage().resize_(act_without_probs_size)
            act_without_probs.untyped_storage().copy_(act_without_probs_.untyped_storage())
            act_without_probs = None
            act_without_probs_.untyped_storage().resize_(0)

        if ctx.use_gmm:
            if config.gemm_gradient_accumulation_fusion:
                grad_weights2 = gmm_cls.op_gmm_add(mm2_inputs, weights2, grad_outs, group_list, original_weight2)
            else:
                grad_weights2 = gmm_cls.op_dw(mm2_inputs, grad_outs, group_list)[0]
        else:
            grad_weights2 = torch.matmul(mm2_inputs.t(), grad_outs)

        # grad of activation_func_with_probs.
        grad_outs.untyped_storage().resize_(0)
        mm2_inputs.untyped_storage().resize_(0)
        act_graph.backward(grad_mm2_inputs)
        permuted_probs_inputs_detach.untyped_storage().resize_(0)
        grad_mm2_inputs.untyped_storage().resize_(0)
        act_inputs.untyped_storage().resize_(0)

        if moe_zero_memory == "level0" or (moe_zero_memory == "level1" and is_only_recompute_activation):
            def alltoall_token_permutation1(hidden_states, routing_map, probs):
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                permutated_local_input_tokens, permuted_probs_, _ = permute(
                    hidden_states, routing_map, probs, num_out_tokens=num_out_tokens, fused=ctx.config.moe_permute_fusion
                )
                return permutated_local_input_tokens, permuted_probs_

            permutated_local_input_tokens, permuted_probs_ = alltoall_token_permutation1(detach_input, routing_map, probs)

            _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
                permutated_local_input_tokens,
                output_splits,
                input_splits,
                ep_group,
            )
            # recompute global_probs.
            _, global_probs, permuted_probs_inputs_handle = async_all_to_all(
                permuted_probs_,
                output_splits,
                input_splits,
                ep_group
            )

        if ctx.use_gmm:
            # Because probs, mm1_inputs_grad consists of two parts.
            mm1_inputs_grad = gmm_cls.op_dx(act_inputs.grad, weights1, group_list)[0]
        else:
            mm1_inputs_grad = torch.matmul(act_inputs.grad, weights1.t())

        probs.untyped_storage().resize_(0)
        # backward for probs. 
        backward_func(permute2_prob_graph, permuted_probs_inputs_detach.grad) 

        _, permute1_prob_backward_input, bw_permute1_prob_all2all_handle = async_all_to_all(
            permute2_prob_detach.grad,
            input_splits,
            output_splits,
            ep_group,
        )

        # backward for expert.
        backward_func(permute2_graph, mm1_inputs_grad)
        mm1_inputs_grad.untyped_storage().resize_(0)

        if moe_zero_memory == "level0" or (moe_zero_memory == "level1" and is_only_recompute_activation):
            permute1_ep_all_to_all_handle.wait()
            permutated_local_input_tokens.untyped_storage().resize_(0)

        _, permute1_backward_input, bw_permute1_ep_all2all_handle = async_all_to_all(
            permute2_input_detach.grad,
            input_splits,
            output_splits,
            ep_group,
        )

        #prepare for permute1 backward. 
        set_all2all_experts_output((permute1_backward_input, bw_permute1_ep_all2all_handle, 
                                    permute1_prob_backward_input, bw_permute1_prob_all2all_handle))

        if moe_zero_memory == "level0" or (moe_zero_memory == "level1" and is_only_recompute_activation):
            permuted_probs_inputs_handle.wait()
            permuted_probs_.untyped_storage().resize_(0)
            mm1_inputs, _ = sort_chunks_by_idxs(
                global_input_tokens,
                num_global_tokens_per_local_expert_cpu.ravel(),
                sort_input_by_local_experts,
                probs=global_probs
            )
            global_probs.untyped_storage().resize_(0)
            global_input_tokens.untyped_storage().resize_(0)

        if ctx.use_gmm:
            if config.gemm_gradient_accumulation_fusion:
                mm1_weights_grad = gmm_cls.op_gmm_add(mm1_inputs, weights1, act_inputs.grad, group_list,
                                                         original_weight1)
            else:
                mm1_weights_grad = gmm_cls.op_dw(mm1_inputs, act_inputs.grad, group_list)[0]
        else:
            mm1_weights_grad = torch.matmul(mm1_inputs.t(), act_inputs.grad)
        act_inputs.grad.untyped_storage().resize_(0)
        permuted_probs_inputs_detach.untyped_storage().resize_(0)
        return mm1_inputs_grad, mm1_weights_grad, grad_weights2, None, None


def grouped_mlp_with_comp_and_comm_overlap_all2allseq(inputs, weights1, weights2, args, ctx):
    return GroupedMlpWithCompAndCommOverlapAll2AllSeq.apply(inputs, weights1, weights2, args, ctx)
