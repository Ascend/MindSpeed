#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import torch
import torch.nn.functional as F
from einops import rearrange
from megatron.training import get_args

from mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules.utils import TensorSwapManager, make_wait_swap_in_hook
from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.core.transformer.moe.grouped_matmul_util import get_gmm_quant_func
from mindspeed.core.transformer.moe.moe_feature import GroupedMLP as MegatronGroupedMLP
from mindspeed.model.transformer import should_recompute_activation
from mindspeed.ops.gmm import GMMFunction
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32
from .weight_grad_store import WeightGradStore


def get_gmm_weight_grad(inputs, grad_out, group_list, group_list_data_type, weight_param, weight_tensor):
    if WeightGradStore.is_decoupleBlock:
        WeightGradStore.put(
            [inputs, group_list, group_list_data_type],
            grad_out,
            weight_param,
            sequence_parallel=False,
            in_row=False,
        )
        if hasattr(weight_param, 'grad_added_to_main_grad') and get_args().overlap_grad_reduce:
            # When overlap_grad_reduce is True, need to ensure that backward hooks
            # are all run on the main backprop thread to prevent deadlocks. Setup
            # dummy grad_weight tensor to prevent backward hooks from being run
            # in a background thread.
            shape = list(weight_tensor.shape)
            shape[1], shape[2] = shape[2], shape[1]
            weight_param.skip_grad_accum = True

        grad_weights = None
    else:
        if get_args().gemm_gradient_accumulation_fusion and not getattr(weight_param, 'is_hot_experts', False):
            npu_groupmatmul_add_fp32(inputs, grad_out, group_list, weight_param.main_grad)
            if hasattr(weight_param, 'grad_added_to_main_grad'):
                shape = list(weight_tensor.shape)
                shape[1], shape[2] = shape[2], shape[1]
                if getattr(weight_tensor, 'zero_out_wgrad', False):
                    grad_weights = torch.zeros(
                        shape,
                        dtype=inputs.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weights = torch.empty(
                        shape,
                        dtype=inputs.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                weight_param.grad_added_to_main_grad = True
            else:
                grad_weights = None
        else:
            grad_weights = GMMFunction.builder.load().npu_gmm([inputs.t()], [grad_out], [], group_list, 2,
                                                              group_list_data_type)[0]

    return grad_weights


class GroupedMatmulWithWeightGradDetach(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight_tensor, weight_param, group_list, in_row=False):
        mm_out = GMMFunction.builder.load().npu_gmm([inputs], [weight_tensor], [], group_list, 0, 0)[0]
        ctx.save_for_backward(inputs, weight_tensor, group_list)
        ctx.weight_param = weight_param
        ctx.in_row = in_row

        return mm_out

    @staticmethod
    def backward(ctx, *grad_outs):
        grad_out = grad_outs[0]
        inputs, weight_tensor, group_list = ctx.saved_tensors
        weight_param = ctx.weight_param
        weight_tensor = rearrange(weight_tensor, 'n h f -> n f h')
        grad_inputs = \
            GMMFunction.builder.load().npu_gmm([grad_out], [weight_tensor], [], group_list, 0, 0)[0]
        grad_weights = get_gmm_weight_grad(inputs, grad_out, group_list, 0, weight_param,
                                           weight_tensor)

        return grad_inputs, grad_weights, None, None, None


def npu_gmm_with_detach(inputs, weight_tensor, weight_param, bias=None, group_list=None):
    quant_gmm_func = get_gmm_quant_func()
    if quant_gmm_func is not None:
        return quant_gmm_func.apply(inputs, weight_tensor, bias, group_list, weight_param)
    return GroupedMatmulWithWeightGradDetach.apply(inputs, weight_tensor, weight_param, group_list)


class MindSpeedFbOverlapGmmExperts(MegatronGroupedMLP):
    # GMM Class for FB Overlap
    def __init__(self, *args, **kwargs):

        super(MindSpeedFbOverlapGmmExperts, self).__init__(*args, **kwargs)
        self.weight1.gmm_weight = True
        self.weight2.gmm_weight = True
        self.layer_number = None
        if self.config.gated_linear_unit:
            assert (self.config.activation_func == F.silu), 'Activation function must be silu when using fused_swiglu.'
            self.activation_func = fused_swiglu

    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs=None):
        args = get_args()
        is_recompute_activation = args.moe_zero_memory != 'disable' or should_recompute_activation(self.layer_number)
        is_swap_fc1 = args.moe_zero_memory == 'level1'

        def act_func(fc1_output, permuted_probs):
            fc2_input = self.activation_func(fc1_output)
            if permuted_probs is not None:
                fc2_input = (fc2_input * permuted_probs.unsqueeze(-1)) \
                    .type(fc2_input.dtype)
            return fc2_input

        fc1_swap_manager = None
        probs_swap_manager = None
        if permuted_local_hidden_states.nelement() != 0:
            group_list = torch.cumsum(tokens_per_expert, dim=0)

            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            fc1_output = npu_gmm_with_detach(permuted_local_hidden_states, w1, self.weight1, bias=None,
                                             group_list=group_list)
            if is_recompute_activation:
                act_ckpt = CheckpointWithoutOutput()
                fc2_input = act_ckpt.checkpoint(act_func, False, fc1_output, permuted_probs)
            else:
                act_ckpt = None
                fc2_input = act_func(fc1_output, permuted_probs)

            if is_swap_fc1:
                TensorSwapManager.wait_all_swap_out("fc1")
                fc1_swap_manager = TensorSwapManager(fc1_output, "fc1")
                fc1_swap_manager.async_swap_out(wait_stream=torch.npu.current_stream())

                TensorSwapManager.wait_all_swap_out("permuted_probs")
                probs_swap_manager = TensorSwapManager(permuted_probs, "permuted_probs")
                probs_swap_manager.async_swap_out(wait_stream=torch.npu.current_stream())

            fc2_output = npu_gmm_with_detach(fc2_input, w2, self.weight2, bias=None, group_list=group_list)
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure parameters still have gradients when no tokens are routed to this set of experts.
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            fc1_output = torch.matmul(permuted_local_hidden_states, w1)
            if is_recompute_activation:
                act_ckpt = CheckpointWithoutOutput()
                fc2_input = act_ckpt.checkpoint(act_func, False, fc1_output, permuted_probs)
            else:
                act_ckpt = None
                fc2_input = act_func(fc1_output, permuted_probs)
            fc2_output = torch.matmul(fc2_input, w2)

        if is_recompute_activation:
            act_ckpt.discard_output()

        return (fc2_output, act_ckpt, fc1_swap_manager, probs_swap_manager), None
