# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from einops import rearrange
import torch
from megatron.training import get_args
from mindspeed.core.pipeline_parallel.fb_overlap.modules.weight_grad_store import WeightGradStore
from mindspeed.ops.gmm import GMMFunction
from mindspeed.model.transformer import should_recompute_activation
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32



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
        if get_args().gemm_gradient_accumulation_fusion:
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
    return GroupedMatmulWithWeightGradDetach.apply(inputs, weight_tensor, weight_param, group_list)



def group_mlp_forward_detach(self, permuted_local_hidden_states, tokens_per_expert):
    args = get_args()
    is_recompute_activation = args.moe_zero_memory == 'level0' or should_recompute_activation(self.layer_number)
    if permuted_local_hidden_states.nelement() != 0:
        group_list = torch.cumsum(tokens_per_expert, dim=0)
        w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
        w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

        fc1_output = npu_gmm_with_detach(permuted_local_hidden_states, w1, self.weight1, bias=None, group_list=group_list)
        intermediate_parallel = self.activation_func(fc1_output)
        fc2_output = npu_gmm_with_detach(intermediate_parallel, w2, self.weight2, bias=None, group_list=group_list)
        if is_recompute_activation:
            intermediate_parallel.untyped_storage().resize_(0)
    else:
        # No token is allocated for local experts.
        assert torch.count_nonzero(tokens_per_expert) == 0

        # Make sure parameters still have gradients when no tokens are routed to this set of experts.
        w1 = self.weight1.view(self.config.hidden_size, -1)
        w2 = self.weight2.view(-1, self.config.hidden_size)
        fc1_output = torch.matmul(permuted_local_hidden_states, w1)
        intermediate_parallel = self.activation_func(fc1_output)
        fc2_output = torch.matmul(intermediate_parallel, w2)
        if is_recompute_activation:
            intermediate_parallel.untyped_storage().resize_(0)


    return (fc2_output, fc1_output, intermediate_parallel), None