# Copyright (c) 2025; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
import torch.nn.functional as F
from mindspeed.core.transformer.moe.moe_feature import parallel_state, get_args, MLP
from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.core.transformer.moe.moe_feature.overlap.grouped_mlp_with_comp_and_comm_overlap_all2all import grouped_mlp_with_comp_and_comm_overlap_all2all
from mindspeed.core.transformer.moe.moe_feature.overlap.grouped_mlp_with_comp_and_comm_overlap_allgather import grouped_mlp_with_comp_and_comm_overlap_allgather


class OverLapGmmExpertsImpl:
    """
    An efficient implementation of the experts layer using GroupedGEMM.
    Only used when open moe_alltoall_overlap_comm or moe_allgather_overlap_comm to overlap compute and communicate.
    """

    def __init__(self, num_local_experts, config=None):
        """
        Args:
            num_local_experts: experts in device
            config: TransformerConfig
        """
        self.num_local_experts = num_local_experts
        self.config = config

        self.weight1 = None
        self.weight2 = None
        self.activation_checkpoint_manager = None
        if self.config.moe_tp_extend_ep:
            tp_size = parallel_state._MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
            # set tp size to 1 before GMM init to aviod weight sharding
            parallel_state._MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = 1
        super().__init__(num_local_experts, config)
        if self.config.moe_tp_extend_ep:
            parallel_state._MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = tp_size
        if self.config.gated_linear_unit:
            assert (self.config.activation_func == F.silu
                    ), 'Activation function must be silu when using fused_swiglu.'
            self.activation_func = fused_swiglu
        self.layer_number = None
        self.set_recompute_activation_func = False
        self.activation_checkpoint_manager = CheckpointWithoutOutput()

        def activation_func_with_probs(x, probs):
            dtype = x.dtype
            res = self.activation_func(x) * probs
            return res.to(dtype)
        self.activation_func_with_probs = activation_func_with_probs

    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs, ctx=None):
        """Forward step of the GroupedMLP with MoE overlap."""

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        if permuted_local_hidden_states.nelement() != 0:
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)
        else:
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
        group_list = torch.cumsum(tokens_per_expert, dim=0)
        if self.config.moe_alltoall_overlap_comm:
            return grouped_mlp_with_comp_and_comm_overlap_all2all(permuted_local_hidden_states, w1, w2,
                                                                (self.weight1, self.weight2, self.activation_func_with_probs,
                                                                permuted_probs, group_list, self.layer_number, self.config),
                                                                ctx=ctx)
        else:
            return grouped_mlp_with_comp_and_comm_overlap_allgather(permuted_local_hidden_states, w1, w2,
                                                                    (self.weight1, self.weight2, self.activation_func_with_probs,
                                                                     permuted_probs, group_list, self.layer_number, self.config))


def Zero_Memory_SharedExpertMlp_forward(self, hidden_states, moe_ctx):
    """Shared expert forward function with zero_memory."""
    output, _ = MLP.forward(self, hidden_states, moe_ctx)
    if self.use_shared_expert_gate:
        logits = torch.nn.functional.linear(hidden_states, self.gate_weight)
        gate_score = torch.nn.functional.sigmoid(logits)
        output = output * gate_score
    return output
