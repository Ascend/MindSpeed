# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import torch
import torch.nn.functional as F
from megatron.core.activations import squared_relu
from megatron.core.fusions.fused_bias_geglu import quick_gelu, weighted_bias_quick_geglu_impl
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection

from mindspeed.args_utils import get_full_args as get_args
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.model.transformer import should_recompute_activation


class MindSpeedFbOverlapGmmExperts(TEGroupedMLP):
    """TE grouped experts with MindSpeed FB-overlap activation lifetime semantics."""

    def __init__(self, *args, **kwargs):
        if kwargs.get("pg_collection") is None:
            kwargs["pg_collection"] = get_default_pg_collection()
        super().__init__(*args, **kwargs)
        self._validate_fb_overlap_config()
        if getattr(self, "_with_fused_impl", False):
            raise RuntimeError(
                "FB-overlap's thin TEGroupedMLP forward requires the two "
                "GroupedLinear children instead of the TE op-fuser path."
            )
        if not (self.linear_fc1.need_backward_dw() and self.linear_fc2.need_backward_dw()):
            raise RuntimeError(
                "FB-overlap level0 releases the FC1 input storage before "
                "backward, so both TE GroupedLinear layers must delay wgrad."
            )
        self.layer_number = None

    def _validate_fb_overlap_config(self):
        """Reject options whose semantics are not represented by the thin FB forward."""
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
        if getattr(self.config, "transformer_impl", None) == "inference_optimized":
            unsupported.append("transformer_impl=inference_optimized")
        if self.offload_expert_fc1:
            unsupported.append("fine_grained_activation_offloading=expert_fc1")
        if self.offload_moe_act:
            unsupported.append("fine_grained_activation_offloading=moe_act")
        if self.activation_recompute:
            unsupported.append("recompute_modules=moe_act")
        if unsupported:
            raise ValueError("MindSpeed FB-overlap TE grouped experts do not support: " + ", ".join(unsupported) + ".")

    def _bias_act_func(self, intermediate_parallel, bias_parallel, permuted_probs):
        """Match Megatron 0.18 TEGroupedMLP's non-fused activation path."""
        if self.config.use_te_activation_func:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)
            if permuted_probs is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * permuted_probs
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        elif self.config.bias_activation_fusion:
            if self.activation_func == F.silu and self.config.gated_linear_unit:
                intermediate_parallel = weighted_bias_swiglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    permuted_probs,
                    self.config.activation_func_fp8_input_store,
                )
            elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                intermediate_parallel = weighted_bias_quick_geglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    permuted_probs,
                    self.config.activation_func_fp8_input_store,
                    self.config.glu_linear_offset,
                    self.config.activation_func_clamp_value,
                )
            else:
                raise ValueError("Only support fusion of swiglu and quick_gelu in TEGroupedMLP.")
        elif self.activation_func == squared_relu and self.config.use_fused_weighted_squared_relu:
            assert bias_parallel is None, "Bias is not supported with fused weighted squared relu."
            intermediate_parallel = weighted_squared_relu_impl(intermediate_parallel, permuted_probs)
        else:
            if self.config.gated_linear_unit:
                x_glu, x_linear = torch.chunk(intermediate_parallel, 2, dim=-1)
                intermediate_parallel = self.config.activation_func(x_glu) * (x_linear + self.config.glu_linear_offset)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)
            original_dtype = intermediate_parallel.dtype
            intermediate_parallel = intermediate_parallel * permuted_probs
            intermediate_parallel = intermediate_parallel.to(original_dtype)
        return intermediate_parallel

    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs=None):
        """Run TE GroupedLinear while preserving the original level0 checkpoint contract."""
        args = get_args()
        if args.moe_zero_memory == "level1":
            raise RuntimeError(
                "MindSpeed FB-overlap with TEGroupedMLP currently supports "
                "moe-zero-memory=disable or level0, but not level1."
            )

        is_recompute_activation = args.moe_zero_memory == "level0" or should_recompute_activation(self.layer_number)
        group_metadata = self.linear_fc1.make_grouped_linear_metadata(tokens_per_expert)
        if permuted_probs is None:
            raise RuntimeError("FB-overlap TE grouped experts require permuted_probs.")
        permuted_probs = permuted_probs.unsqueeze(-1)

        # Keep the numerical path identical to TEGroupedMLP.forward. In
        # particular, do not force npu_swiglu here: TEGroupedMLP selects its
        # fused or unfused activation backend from the same config in both the
        # overlap and non-overlap cases.
        if self.config.moe_apply_probs_on_input:
            assert self.config.moe_router_topk == 1, "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            permuted_probs = torch.ones_like(permuted_probs)

        # Keep this forward intentionally thin. TE owns both GroupedLinear
        # autograd functions and their delayed wgrad queues; MindSpeed keeps
        # ownership of activation recomputation and the outer communication
        # schedule used to restore the FC1 input before backward_dw().
        fc1_output, fc1_bias = self.linear_fc1(permuted_local_hidden_states, group_metadata)
        if fc1_bias is not None:
            raise RuntimeError("FB-overlap TE grouped experts do not support FC1 bias.")

        if is_recompute_activation:
            act_ckpt = CheckpointWithoutOutput()
            fc2_input = act_ckpt.checkpoint(
                self._bias_act_func,
                False,
                fc1_output,
                fc1_bias,
                permuted_probs,
            )
        else:
            act_ckpt = None
            fc2_input = self._bias_act_func(fc1_output, fc1_bias, permuted_probs)

        fc2_output, fc2_bias = self.linear_fc2(fc2_input, group_metadata)
        if fc2_bias is not None:
            raise RuntimeError("FB-overlap TE grouped experts do not support FC2 bias.")

        if is_recompute_activation:
            act_ckpt.discard_output()

        # level0 has no swap managers. Keep the existing scheduler-facing
        # return contract unchanged.
        return (fc2_output, act_ckpt, None, None), None
