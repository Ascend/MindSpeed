# Copyright (c) 2025; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

# This implementation is composed with GroupedMLP at runtime, so pylint cannot
# see the weight attributes initialized by the cooperative super() call.
# pylint: disable=no-member

import torch
import torch.nn.functional as F
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from mindspeed.core.transformer.moe.moe_feature import parallel_state, MLP
from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.core.transformer.moe.moe_feature.overlap.grouped_mlp_with_comp_and_comm_overlap_all2allseq import (
    grouped_mlp_with_comp_and_comm_overlap_all2allseq,
)
from mindspeed.core.transformer.moe.moe_feature.overlap.grouped_mlp_with_comp_and_comm_overlap_allgather import (
    grouped_mlp_with_comp_and_comm_overlap_allgather,
)
from mindspeed.model.transformer import should_recompute_activation


class OverLapGmmExpertsImpl:
    """
    An efficient implementation of the experts layer using GroupedGEMM.
    Only used when open moe_alltoall_overlap_comm or moe_allgather_overlap_comm to overlap compute and communicate.
    """

    def __init__(self, num_local_experts, config=None, pg_collection=None, name=None, **kwargs):
        """
        Args:
            num_local_experts: experts in device
            config: TransformerConfig
            **kwargs: Capture Megatron 0.18 builder params (submodules, etc.)
        """
        self.num_local_experts = num_local_experts
        self.config = config

        self.activation_checkpoint_manager = None
        if self.config.moe_tp_extend_ep:
            tp_size = parallel_state._MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
            # set tp size to 1 before GMM init to avoid weight sharding
            parallel_state._MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = 1
        super().__init__(
            num_local_experts,
            config,
            pg_collection=pg_collection,
            name=name,
            **kwargs,
        )
        if self.config.moe_tp_extend_ep:
            parallel_state._MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = tp_size
        if self.config.gated_linear_unit:
            assert self.config.activation_func == F.silu, 'Activation function must be silu when using fused_swiglu.'
            self.activation_func = fused_swiglu
        self.layer_number = None
        self.set_recompute_activation_func = False
        self.activation_checkpoint_manager = CheckpointWithoutOutput()

    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs, ctx=None):
        """Forward step of the GroupedMLP with MoE overlap."""

        if self.config.moe_apply_probs_on_input:
            assert self.config.moe_router_topk == 1, "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
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
            return grouped_mlp_with_comp_and_comm_overlap_all2allseq(
                permuted_local_hidden_states,
                w1,
                w2,
                (
                    self.weight1,
                    self.weight2,
                    self.activation_func,
                    permuted_probs,
                    group_list,
                    self.layer_number,
                    self.config,
                ),
                ctx=ctx,
            )
        else:
            return grouped_mlp_with_comp_and_comm_overlap_allgather(
                permuted_local_hidden_states,
                w1,
                w2,
                (self.weight1, self.weight2, self.activation_func, group_list, self.layer_number, self.config),
            )


class AlltoAllOverLapGmmExpertsImpl(TEGroupedMLP):
    """
    TE grouped experts used by alltoall communication overlap.

    The outer MoE autograd function owns communication scheduling. This class
    intentionally keeps only the expert MLP forward and activation checkpoint
    needed by moe-zero-memory level0.
    """

    def __init__(self, num_local_experts, config=None, pg_collection=None, name=None, **kwargs):
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super().__init__(
            num_local_experts,
            config=config,
            pg_collection=pg_collection,
            name=name,
            **kwargs,
        )
        self._validate_alltoall_overlap_config()
        if getattr(self, "_with_fused_impl", False):
            raise RuntimeError(
                "AlltoAll overlap's thin TEGroupedMLP forward requires the "
                "two GroupedLinear children instead of the TE op-fuser path."
            )
        if not (
            getattr(self.linear_fc1, "delay_wgrad_compute", False)
            and getattr(self.linear_fc2, "delay_wgrad_compute", False)
        ):
            raise RuntimeError(
                "AlltoAll overlap level0 releases the FC1 input storage before "
                "backward, so both TE GroupedLinear layers must delay wgrad."
            )
        if self.config.gated_linear_unit:
            assert self.config.activation_func == F.silu, "Activation function must be silu when using fused_swiglu."
            self.activation_func = fused_swiglu
        self.layer_number = None

    def _validate_alltoall_overlap_config(self):
        """Reject options not represented by the thin alltoall-overlap forward."""
        unsupported = []
        if self.config.add_bias_linear:
            unsupported.append("add_bias_linear")
        if getattr(self.config, "moe_latent_size", None) is not None:
            unsupported.append("moe_latent_size")
        if getattr(self.config, "moe_mlp_glu_interleave_size", None) is not None:
            unsupported.append("moe_mlp_glu_interleave_size")
        if getattr(self.config, "glu_linear_offset", 0.0) != 0.0:
            unsupported.append("glu_linear_offset")
        if getattr(self.config, "activation_func_clamp_value", None) is not None:
            unsupported.append("activation_func_clamp_value")
        if getattr(self.config, "use_te_activation_func", False):
            unsupported.append("use_te_activation_func")
        if getattr(self.config, "fp8", None):
            unsupported.append("fp8")
        if getattr(self.config, "fp4", None):
            unsupported.append("fp4")
        if getattr(self.config, "fine_grained_activation_offloading", False):
            unsupported.append("fine_grained_activation_offloading")
        if getattr(self.config, "transformer_impl", None) == "inference_optimized":
            unsupported.append("transformer_impl=inference_optimized")
        if self.config.moe_zero_memory == "level1":
            unsupported.append("moe_zero_memory=level1")
        if self.config.moe_zero_memory == "level0" and self.config.moe_apply_probs_on_input:
            unsupported.append("moe_apply_probs_on_input with moe_zero_memory=level0")
        if unsupported:
            raise ValueError(
                "MindSpeed alltoall-overlap TE grouped experts do not support: " + ", ".join(unsupported) + "."
            )

    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs, ctx=None):
        """
        Run the two TE GroupedLinear layers and return the level0 activation checkpoint.

        The checkpoint is consumed by MoELayerOverlapAllToAll.backward before
        entering TE backward. The outer scheduler also supplies recomputed FC1
        and FC2 inputs to their delayed-wgrad stores.
        """
        if self.config.moe_apply_probs_on_input:
            assert self.config.moe_router_topk == 1, "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        group_metadata = self.linear_fc1.make_grouped_linear_metadata(tokens_per_expert)
        fc1_output, fc1_bias = self.linear_fc1(permuted_local_hidden_states, group_metadata)
        if fc1_bias is not None:
            raise RuntimeError("AlltoAll-overlap TE grouped experts do not support FC1 bias.")

        if self.config.moe_zero_memory == "level0":
            permuted_local_hidden_states.untyped_storage().resize_(0)

        def activation_func_with_probs(fc1_input, probs):
            fc2_input = self.activation_func(fc1_input)
            if probs is not None:
                original_dtype = fc2_input.dtype
                fc2_input = fc2_input * probs.unsqueeze(-1)
                fc2_input = fc2_input.to(original_dtype)
            return fc2_input

        is_recompute_activation = self.config.moe_zero_memory == "level0" or should_recompute_activation(
            self.layer_number
        )
        if is_recompute_activation:
            activation_checkpoint = CheckpointWithoutOutput()
            fc2_input = activation_checkpoint.checkpoint(
                activation_func_with_probs,
                False,
                fc1_output,
                permuted_probs,
            )
        else:
            activation_checkpoint = None
            fc2_input = activation_func_with_probs(fc1_output, permuted_probs)

        fc2_output, fc2_bias = self.linear_fc2(fc2_input, group_metadata)
        if fc2_bias is not None:
            raise RuntimeError("AlltoAll-overlap TE grouped experts do not support FC2 bias.")

        if activation_checkpoint is not None:
            activation_checkpoint.discard_output()

        return (fc2_output, None), activation_checkpoint


def zero_memory_shared_expert_mlp_forward(self, hidden_states, moe_ctx):
    """Shared expert forward function with zero_memory."""
    output, _ = MLP.forward(self, hidden_states, moe_ctx=moe_ctx)
    if self.use_shared_expert_gate:
        logits = torch.nn.functional.linear(hidden_states, self.gate_weight)
        gate_score = torch.nn.functional.sigmoid(logits)
        output = output * gate_score
    return output
