# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""
GroupedMLP: NPU-optimized grouped MLP for Megatron 0.18 MoE experts.

This class provides the weight layout used by MindSpeed expert
implementations: self.weight1, self.weight2, self.num_local_experts and
self.config.

Key differences from Megatron's GroupedMLP:
  - Implements Megatron's ExpertsBuilder interface, including explicit
    process-group and module-name propagation.
  - Uses MindSpeed's own grouped_gemm_util.ops.gmm() which wraps
    torch_npu.npu_grouped_matmul() for NPU acceleration.
  - Implements the Megatron 0.18 ExpertsInterface backward_dw() hook.
"""

from functools import partial

import torch
from torch.nn.parameter import Parameter

from megatron.core import parallel_state, utils
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from mindspeed.core.transformer.moe import grouped_gemm_util as gg


class GroupedMLP(MegatronModule):
    """NPU-optimized Grouped MLP for MoE experts.

    Serves as the base class for MindSpeed MoE expert implementations running
    against Megatron 0.18.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        *,
        pg_collection=None,
        name=None,
        **kwargs,
    ):
        super().__init__(config=config)
        self.submodules = kwargs.pop('submodules', None)
        if kwargs:
            unexpected = ', '.join(sorted(kwargs))
            raise TypeError(f'GroupedMLP got unexpected keyword arguments: {unexpected}')
        self._validate_supported_config(config)

        self.num_local_experts = num_local_experts
        self.pg_collection = pg_collection
        self.name = name
        self.ep_group = getattr(pg_collection, 'ep', None)
        self.tp_group = getattr(pg_collection, 'expt_tp', None)
        self.expt_tp_group = self.tp_group
        self.expt_dp_group = getattr(pg_collection, 'expt_dp', None)
        gg.assert_grouped_gemm_is_available()

        self.expert_parallel = (
            utils.get_pg_size(self.ep_group) > 1 if self.ep_group is not None else config.expert_model_parallel_size > 1
        )

        # Activation function — default from config.
        # Subclasses (GmmExpertsImpl, MindSpeedFbOverlapGmmExperts, etc.)
        # typically override this with fused_swiglu or NPU-optimized variants.
        self.activation_func = self.config.activation_func

        # Compute TP-aware weight shapes (matching old GroupedMLP logic).
        tp_size = (
            utils.get_pg_size(self.tp_group)
            if self.tp_group is not None
            else parallel_state.get_expert_tensor_parallel_world_size()
        )

        fc1_output_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        if config.use_cpu_initialization:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                tp_rank = (
                    utils.get_pg_rank(self.tp_group)
                    if self.tp_group is not None
                    else parallel_state.get_expert_tensor_parallel_rank()
                )
                _initialize_affine_weight_cpu(
                    self.weight1,
                    self.config.hidden_size,
                    fc1_output_size,
                    fc1_output_size_per_partition,
                    partition_dim=1,
                    init_method=config.init_method,
                    params_dtype=config.params_dtype,
                    rank=tp_rank,
                    world_size=tp_size,
                )
                _initialize_affine_weight_cpu(
                    self.weight2,
                    fc2_input_size,
                    self.config.hidden_size,
                    fc2_input_size_per_partition,
                    partition_dim=0,
                    init_method=config.output_layer_init_method,
                    params_dtype=config.params_dtype,
                    rank=tp_rank,
                    world_size=tp_size,
                )
        else:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    device=torch.npu.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    device=torch.npu.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(self.weight1, config.init_method, partition_dim=1, is_expert=True)
                _initialize_affine_weight_gpu(
                    self.weight2, config.output_layer_init_method, partition_dim=0, is_expert=True
                )

        setattr(self.weight1, 'allreduce', not self.expert_parallel)
        setattr(self.weight2, 'allreduce', not self.expert_parallel)

    @staticmethod
    def _validate_supported_config(config: TransformerConfig):
        """Reject options that the basic NPU GMM path cannot represent safely."""
        unsupported = []
        if config.add_bias_linear:
            unsupported.append('add_bias_linear')
        if getattr(config, 'moe_latent_size', None) is not None:
            unsupported.append('moe_latent_size')
        if getattr(config, 'moe_mlp_glu_interleave_size', None) is not None:
            unsupported.append('moe_mlp_glu_interleave_size')
        if getattr(config, 'glu_linear_offset', 0.0) != 0.0:
            unsupported.append('glu_linear_offset')
        if getattr(config, 'activation_func_clamp_value', None) is not None:
            unsupported.append('activation_func_clamp_value')
        if getattr(config, 'use_te_activation_func', False):
            unsupported.append('use_te_activation_func')
        if getattr(config, 'moe_single_grouped_weight', False):
            unsupported.append('moe_single_grouped_weight')
        if getattr(config, 'moe_single_grouped_bias', False):
            unsupported.append('moe_single_grouped_bias')
        if getattr(config, 'fp8', None):
            unsupported.append('fp8')
        if getattr(config, 'fp4', None):
            unsupported.append('fp4')
        if getattr(config, 'delay_wgrad_compute', False):
            unsupported.append('delay_wgrad_compute')
        if getattr(config, 'overlap_dispatch_backward_with_experts_wgrad', False):
            unsupported.append('overlap_dispatch_backward_with_experts_wgrad')
        if getattr(config, 'transformer_impl', None) == 'inference_optimized':
            unsupported.append('transformer_impl=inference_optimized')

        if unsupported:
            raise ValueError(
                'The basic MindSpeed NPU grouped-GEMM experts do not support: '
                + ', '.join(unsupported)
                + '. Disable these options or use Megatron/TE grouped experts.'
            )

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor = None,
    ):
        """Default forward path using NPU grouped GEMM.

        fb_overlap subclasses override this with their own forward that
        includes WeightGradStore decoupling and memory optimization.

        Returns:
            tuple: (fc2_output, None) matching the old GroupedMLP signature.
        """
        if permuted_local_hidden_states.nelement() != 0:
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            fc1_output = gg.ops.gmm(permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False)

            if permuted_probs is not None:
                intermediate_parallel = self.activation_func(fc1_output)
                intermediate_parallel = (intermediate_parallel * permuted_probs.unsqueeze(-1)).to(
                    intermediate_parallel.dtype
                )
            else:
                intermediate_parallel = self.activation_func(fc1_output)

            fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
        else:
            assert torch.count_nonzero(tokens_per_expert) == 0
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            if permuted_probs is not None:
                h = self.activation_func(h)
                h = (h * permuted_probs.unsqueeze(-1)).to(h.dtype)
            else:
                h = self.activation_func(h)
            fc2_output = torch.matmul(h, w2)

        return fc2_output, None

    def backward_dw(self):
        """Satisfy the Megatron ExpertsInterface for the basic autograd path.

        In standard training, weight gradients are computed automatically
        by PyTorch autograd and this is a no-op. Delayed weight-gradient
        modes are rejected during initialization.
        """
        pass

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Distributed checkpoint sharding compatible with sequential experts."""
        from megatron.core.dist_checkpointing.mapping import (
            ReplicaId,
            ShardedTensor,
            ShardedTensorFactory,
        )

        sharded_state_dict = {}
        ep_size = (
            self.ep_group.size() if self.ep_group is not None else parallel_state.get_expert_model_parallel_world_size()
        )
        ep_rank = self.ep_group.rank() if self.ep_group is not None else parallel_state.get_expert_model_parallel_rank()
        tp_size = (
            self.tp_group.size()
            if self.tp_group is not None
            else parallel_state.get_expert_tensor_parallel_world_size()
        )
        tp_rank = (
            self.tp_group.rank() if self.tp_group is not None else parallel_state.get_expert_tensor_parallel_rank()
        )
        dp_rank = (
            self.expt_dp_group.rank()
            if self.expt_dp_group is not None
            else parallel_state.get_expert_data_parallel_rank()
        )
        prepend_axis_num = len(sharded_offsets)
        replica_id = (0, 0, dp_rank)

        local_ffn_dim_size = self.weight2.numel() // self.num_local_experts // self.config.hidden_size

        @torch.no_grad()
        def sh_ten_build_fn(
            key: str,
            t: torch.Tensor,
            replica_id: ReplicaId,
            flattened_range: slice,
            tp_axis: int,
            with_glu: bool,
        ):
            if tp_axis == 1:
                last_dim_size = local_ffn_dim_size * 2 if with_glu else local_ffn_dim_size
                real_shape = (self.num_local_experts, self.config.hidden_size, last_dim_size)
            elif tp_axis == 0:
                real_shape = (self.num_local_experts, local_ffn_dim_size, self.config.hidden_size)
            else:
                raise ValueError("tp_axis should be 0 or 1.")
            if flattened_range is not None:
                raise RuntimeError(
                    'Megatron 0.18 no longer supports flattened ShardedTensor ranges; '
                    'GroupedMLP checkpointing cannot save a flattened optimizer state.'
                )

            t = t.view(real_shape).transpose(-1, -2)
            tp_axis = 1 - tp_axis
            if with_glu:
                local_tensors = torch.chunk(t, 2, -2)
                return [
                    ShardedTensor.from_rank_offsets(
                        key,
                        local_tensors[0].contiguous(),
                        *sharded_offsets,
                        (prepend_axis_num, ep_rank, ep_size),
                        (prepend_axis_num + 1, tp_rank, tp_size * 2),
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                    ),
                    ShardedTensor.from_rank_offsets(
                        key,
                        local_tensors[1].contiguous(),
                        *sharded_offsets,
                        (prepend_axis_num, ep_rank, ep_size),
                        (prepend_axis_num + 1, tp_size + tp_rank, tp_size * 2),
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                    ),
                ]
            return ShardedTensor.from_rank_offsets(
                key,
                t.contiguous(),
                *sharded_offsets,
                (prepend_axis_num, ep_rank, ep_size),
                (prepend_axis_num + 1 + tp_axis, tp_rank, tp_size),
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            )

        @torch.no_grad()
        def sh_ten_merge_fn(sub_state_dict, tp_axis: int, with_glu: bool):
            if tp_axis == 1:
                weight_shape = (self.config.hidden_size, -1)
            elif tp_axis == 0:
                weight_shape = (-1, self.config.hidden_size)
            else:
                raise ValueError("tp_axis should be 0 or 1.")
            if with_glu:
                sub_state_dict = torch.cat(sub_state_dict, -2)
            return sub_state_dict.transpose(-1, -2).reshape(weight_shape)

        state_dict = self.state_dict(prefix='', keep_vars=True)
        for name, tensor in state_dict.items():
            if name == 'weight1':
                tp_axis = 1
                with_glu = self.config.gated_linear_unit
                checkpoint_key = f'{prefix}experts.linear_fc1.weight'
            elif name == 'weight2':
                tp_axis = 0
                with_glu = False
                checkpoint_key = f'{prefix}experts.linear_fc2.weight'
            else:
                continue

            sharded_state_dict[f'{prefix}{name}'] = ShardedTensorFactory(
                checkpoint_key,
                tensor,
                partial(sh_ten_build_fn, tp_axis=tp_axis, with_glu=with_glu),
                partial(sh_ten_merge_fn, tp_axis=tp_axis, with_glu=with_glu),
                replica_id,
            )

        return sharded_state_dict

    def set_layer_number(self, layer_number: int):
        """Set the layer number — called by transformer_block init wrappers."""
        self.layer_number = layer_number
