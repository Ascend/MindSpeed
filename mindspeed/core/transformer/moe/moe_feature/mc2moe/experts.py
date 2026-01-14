# Copyright (c) 2025, Huawei Technologies.
# All rights reserved.

import torch
import torch.nn.functional as F
from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.core.transformer.moe.moe_feature import grouped_gemm_util as gg
from mindspeed.model.transformer import should_recompute_activation
from mindspeed.core.transformer.moe.moe_feature import parallel_state
from .mc2_fuse_a2a import AlltoallvPermuteGmm, GmmUnpermuteAlltoallv


class GmmExpertsMC2Impl:
    """An efficient implementation of the Experts layer using GroupedGEMM.

    Executes multiple experts in parallel to maximize computational efficiency.
    support gemm_fusion and activation recompute.
    """
    def __init__(self, num_local_experts, config=None):
        """adjust the logic for generate expert weight to avoid splitting by tp_size

        Args:
            num_local_experts: experts in device
            config: TransformerConfig
        """
        self.num_local_experts = num_local_experts
        self.config = config

        # use Megatron GroupedMLP to init to get params
        self.layer_number = None
        self.weight1 = None
        self.weight2 = None
        super().__init__(num_local_experts, config)
        if self.config.gated_linear_unit:
            assert (self.config.activation_func == F.silu), 'Activation function must be silu when using fused_swiglu.'
            self.activation_func = fused_swiglu

    def forward(self, permuted_local_hidden_states_af_weight1, tokens_per_expert, permuted_probs=None, moe_layer_dispatcher=None):
        """Forward of GroupedMLP

        Args:
            permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the
            local experts.
            tokens_per_expert (torch.Tensor): The number of tokens per expert.
            permuted_probs (torch.Tensor): Permuted Probs for each expert.
            routing_map (torch.Tensor): The mapping of tokens to experts, with shape
                [num_tokens, num_experts].
            moe_layer_dispatcher: The MoEAlltoAllMC2TokenDispatcher.

        Return:
            output (torch.Tensor): The output of the local experts.\

        Warning:Due to kernal BUG, shared_expert in gmmmc2 is unused now.
                With shared_expert, Megatron's Shared_expert_overlap will work.

        """
        import sys
        is_recompute_activation = should_recompute_activation(
            self.layer_number) and not self.config.moe_alltoall_overlap_comm and not \
                                      self.config.moe_allgather_overlap_comm

        w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
        w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

        mm1_out, shared_expert_mm1_out = AlltoallvPermuteGmm.apply(permuted_local_hidden_states_af_weight1, 
                                                                    w1,  
                                                                    parallel_state.get_expert_model_parallel_group(), 
                                                                    tokens_per_expert, 
                                                                    None,
                                                                    None,
                                                                    None,
                                                                    moe_layer_dispatcher.send_counts,
                                                                    moe_layer_dispatcher.recv_counts
                                                                    )

        if not is_recompute_activation:
            intermediate_parallel = self.activation_func_with_probs(mm1_out, permuted_probs.unsqueeze(-1))
        else:
            self.activation_checkpoint_manager = CheckpointWithoutOutput()
            intermediate_parallel = self.activation_checkpoint_manager.checkpoint(self.activation_func_with_probs,
                                                                                    False,
                                                                                    mm1_out, permuted_probs.unsqueeze(-1))

        if is_recompute_activation:
            # discard the output of the activation function,
            # which will be restored by recomputation during backward.
            self.activation_checkpoint_manager.discard_output()

            # when backward to output of dense_4h_to_h,
            # recompute and restore the output of activation function.
            if intermediate_parallel.requires_grad:
                intermediate_parallel.register_hook(self.activation_checkpoint_manager.recompute)

        up_out, shared_expert_mm2_out = GmmUnpermuteAlltoallv.apply(intermediate_parallel, 
                                                w2, 
                                                parallel_state.get_expert_model_parallel_group(), 
                                                tokens_per_expert, 
                                                None,
                                                None,
                                                None,
                                                moe_layer_dispatcher.send_counts,
                                                moe_layer_dispatcher.recv_counts)

        return up_out, None
