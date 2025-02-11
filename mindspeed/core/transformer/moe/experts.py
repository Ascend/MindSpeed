# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from functools import wraps
import torch
import torch.nn.functional as F
from megatron.core import parallel_state, tensor_parallel
from megatron.training import get_args
from mindspeed.model.transformer import should_recompute_activation
from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.core.transformer.moe.grouped_gemm_util import fused_alltoall_gather_bmm, fused_bmm_reducescatter_alltoall
from mindspeed.core.transformer.moe.grouped_mlp_with_comp_and_comm_overlap_all2all import grouped_mlp_with_comp_and_comm_overlap_all2all
from mindspeed.core.transformer.moe.grouped_mlp_with_comp_and_comm_overlap_allgather import grouped_mlp_with_comp_and_comm_overlap_allgather
from mindspeed.core.transformer.moe import grouped_gemm_util as gg


def get_zeros_with_tp(input_):
    world_size = parallel_state.get_tensor_model_parallel_world_size()
    zeros_shape = input_.shape[:-1] + (input_.shape[-1] * world_size,)
    return torch.zeros(zeros_shape, dtype=input_.dtype, layout=input_.layout, device=input_.device)


def sequential_mlp_forward(self, permuted_local_hidden_states, tokens_per_expert):
    output_local = get_zeros_with_tp(permuted_local_hidden_states)
    output_bias_local = None
    if self.add_bias:
        output_bias_local = get_zeros_with_tp(permuted_local_hidden_states)

    cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
    # Insert zero at the begining for offset index's convenience
    zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)
    cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))

    if parallel_state.get_tensor_model_parallel_world_size() > 1:
        if not hasattr(self, 'comm_stream'):
            self.comm_stream = torch.cuda.Stream()
        self.comm_stream.wait_stream(torch.cuda.current_stream())

    for expert_num, expert in enumerate(self.local_experts):
        start = cumsum_num_tokens[expert_num]
        end = cumsum_num_tokens[expert_num + 1]
        hidden = permuted_local_hidden_states[start:end]

        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            with torch.cuda.stream(self.comm_stream):
                hidden = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(hidden)
            torch.cuda.current_stream().wait_stream(self.comm_stream)

        output, output_bias = expert(hidden)

        output_local[start:end] = output
        if self.add_bias:
            output_bias = output_bias.expand_as(output)
            output_bias_local[start:end, :] = output_bias

    return output_local, output_bias_local


def group_mlp_forward(self, permuted_local_hidden_states, tokens_per_expert, ctx=None):
    if permuted_local_hidden_states.nelement() != 0:
        w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
        w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)
    else:
        w1 = self.weight1.view(self.config.hidden_size, -1)
        w2 = self.weight2.view(-1, self.config.hidden_size)
    group_list = torch.cumsum(tokens_per_expert, dim=0)
    if get_args().moe_alltoall_overlap_comm:
        return grouped_mlp_with_comp_and_comm_overlap_all2all(permuted_local_hidden_states, w1, w2,
                                                              (self.weight1, self.weight2, self.activation_func,
                                                               group_list, ctx.layer_number),
                                                              ctx=ctx)
    else:
        return grouped_mlp_with_comp_and_comm_overlap_allgather(permuted_local_hidden_states, w1, w2,
                                                                (self.weight1, self.weight2, self.activation_func,
                                                                 group_list, self.layer_number))


def groupedmlp_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        args_ = get_args()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        # set tp size to 1 before GMM init to aviod weight sharding
        if args_.moe_tp_extend_ep:
            parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1
        fn(self, *args, **kwargs)
        if args_.moe_tp_extend_ep:
            parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = tp_size
        if self.config.gated_linear_unit:
            assert (self.config.activation_func == F.silu
                ), 'Activation function must be silu when using fused_swiglu.'
            self.activation_func = fused_swiglu
        self.layer_number = None
        self.set_recompute_activation_func = False

    return wrapper


def groupedmlp_forward(self, permuted_local_hidden_states, tokens_per_expert):
    args = get_args()
    is_recompute_activation = should_recompute_activation(
        self.layer_number) and not args.moe_alltoall_overlap_comm and not args.moe_allgather_overlap_comm
    
    gemm_fusion = args.gemm_gradient_accumulation_fusion
    tp_group = parallel_state.get_tensor_model_parallel_group()
    ep_group = parallel_state.get_expert_model_parallel_group()

    if not is_recompute_activation:
        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            if args.moe_bmm_mc2:
                # input to alltoall_gather_bmm op input: [E*C, H/TP] -> [E, C, H/TP]
                permuted_local_hidden_states = permuted_local_hidden_states.view(self.config.num_moe_experts,
                                                                                 permuted_local_hidden_states.shape[
                                                                                     0] // self.config.num_moe_experts,
                                                                                 -1)
                bmm_param = {'group_ep': ep_group, 'group_tp': tp_group, 'shard_type': 0,
                             'need_recompute': False}
                fc1_output = fused_alltoall_gather_bmm(permuted_local_hidden_states, w1, None, bmm_param)
                intermediate_parallel = self.activation_func(fc1_output)
                fc2_output = fused_bmm_reducescatter_alltoall(intermediate_parallel, w2, None, bmm_param)
                # revert the output shape: [E, C, H/TP] -> [E*C, H/TP]
                fc2_output = fc2_output.view(-1, fc2_output.shape[2])
            else:
                fc1_output = gg.ops.gmm(permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False,
                                        gemm_fusion=gemm_fusion, original_weight=self.weight1)
                intermediate_parallel = self.activation_func(fc1_output)
                fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False,
                                        gemm_fusion=gemm_fusion, original_weight=self.weight2)
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure parameters still have gradients when no tokens are routed to this set of experts.
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            h = self.activation_func(h)
            h = torch.matmul(h, w2)
            fc2_output = h
    else:
        if permuted_local_hidden_states.nelement() != 0:
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            bmm_param = {'group_ep': ep_group, 'group_tp': tp_group, 'shard_type': 0,
                         'need_recompute': False}

            if args.moe_bmm_mc2:
                # input to alltoall_gather_bmm op input: [E*C, H/TP] -> [E, C, H/TP]
                permuted_local_hidden_states = permuted_local_hidden_states.view(self.config.num_moe_experts,
                                                                                 permuted_local_hidden_states.shape[
                                                                                     0] // self.config.num_moe_experts,
                                                                                 -1)

                fc1_output = fused_alltoall_gather_bmm(permuted_local_hidden_states, w1, None, bmm_param)
            else:
                fc1_output = gg.ops.gmm(
                    permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False, gemm_fusion=gemm_fusion,
                    original_weight=self.weight1
                )

            self.activation_checkpoint_manager = CheckpointWithoutOutput()
            intermediate_parallel = self.activation_checkpoint_manager.checkpoint(self.activation_func,
                                                                                  False,
                                                                                  fc1_output)
            if args.moe_bmm_mc2:
                fc2_output = fused_bmm_reducescatter_alltoall(intermediate_parallel, w2, None, bmm_param)
                # revert the output shape: [E, C, H/TP] -> [E*C, H/TP]
                fc2_output = fc2_output.view(-1, fc2_output.shape[2])
            else:
                fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False,
                                        gemm_fusion=gemm_fusion, original_weight=self.weight2)
        else:
            assert torch.count_nonzero(tokens_per_expert) == 0
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            self.activation_checkpoint_manager = CheckpointWithoutOutput()
            intermediate_parallel = self.activation_checkpoint_manager.checkpoint(self.activation_func,
                                                                                  False,
                                                                                  h)
            h = torch.matmul(intermediate_parallel, w2)
            fc2_output = h

        # discard the output of the activation function,
        # which will be restored by recomputation during backward.
        self.activation_checkpoint_manager.discard_output()

        # when backward to output of dense_4h_to_h,
        # recompute and restore the output of activation function.
        if fc2_output.requires_grad:
            fc2_output.register_hook(self.activation_checkpoint_manager.recompute)
    return fc2_output, None
