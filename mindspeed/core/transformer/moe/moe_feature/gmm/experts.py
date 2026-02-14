# Copyright (c) 2025, Huawei Technologies.
# All rights reserved.
import torch
import torch.nn.functional as F
from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.core.transformer.moe.moe_feature import grouped_gemm_util as gg
from mindspeed.model.transformer import should_recompute_activation


class GmmExpertsImpl:
    """An efficient implementation of the Experts layer using GroupedGEMM.

    Executes multiple experts in parallel to maximize computational efficiency.
    support gemm_fusion and activation recompute.
    """

    def __init__(self, num_local_experts, config=None, pg_collection=None):
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
        super().__init__(num_local_experts, config, pg_collection)
        if self.config.gated_linear_unit:
            assert (self.config.activation_func == F.silu), 'Activation function must be silu when using fused_swiglu.'
            self.activation_func = fused_swiglu

    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs=None):
        """Forward of GroupedMLP

        Args:
            permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the
            local experts.
            tokens_per_expert (torch.Tensor): The number of tokens per expert.

        Return:
            output (torch.Tensor): The output of the local experts.
        """
        is_recompute_activation = should_recompute_activation(
            self.layer_number) and not self.config.moe_alltoall_overlap_comm and not \
                                      self.config.moe_allgather_overlap_comm

        gemm_fusion = self.config.gemm_gradient_accumulation_fusion

        if permuted_local_hidden_states.nelement() != 0:
            from mindspeed.core.transformer.moe.grouped_matmul_util import get_gmm_quant_func
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)
            quant_gmm_func = get_gmm_quant_func()
            if quant_gmm_func:
                fc1_output = quant_gmm_func.gmm_apply(permuted_local_hidden_states, w1, None, tokens_per_expert)
            else:
                fc1_output = gg.ops.gmm(
                    permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False, gemm_fusion=gemm_fusion,
                    original_weight=self.weight1
                )
            if not is_recompute_activation:
                intermediate_parallel = self.activation_func_with_probs(fc1_output, permuted_probs.unsqueeze(-1))
            else:
                self.activation_checkpoint_manager = CheckpointWithoutOutput()
                intermediate_parallel = self.activation_checkpoint_manager.checkpoint(self.activation_func_with_probs,
                                                                                      False,
                                                                                      fc1_output,
                                                                                      permuted_probs.unsqueeze(-1))
            if quant_gmm_func:
                fc2_output = quant_gmm_func.gmm_apply(intermediate_parallel, w2, None, tokens_per_expert)
            else:
                fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False,
                                        gemm_fusion=gemm_fusion, original_weight=self.weight2)
        else:
            assert torch.count_nonzero(tokens_per_expert) == 0
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            if not is_recompute_activation:
                intermediate_parallel = self.activation_func_with_probs(h, permuted_probs.unsqueeze(-1))
            else:
                self.activation_checkpoint_manager = CheckpointWithoutOutput()
                intermediate_parallel = self.activation_checkpoint_manager.checkpoint(self.activation_func_with_probs,
                                                                                      False,
                                                                                      h, permuted_probs.unsqueeze(-1))
            h = torch.matmul(intermediate_parallel, w2)
            fc2_output = h

        if is_recompute_activation:
            # discard the output of the activation function,
            # which will be restored by recomputation during backward.
            self.activation_checkpoint_manager.discard_output()

            # when backward to output of dense_4h_to_h,
            # recompute and restore the output of activation function.
            if fc2_output.requires_grad:
                fc2_output.register_hook(self.activation_checkpoint_manager.recompute)

        return fc2_output, None
