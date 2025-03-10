# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from dataclasses import make_dataclass, field
from functools import wraps

import torch.nn.functional as F

from megatron.core.transformer import TransformerConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from megatron.training import get_args


def transformer_config_post_init(self):
    super(TransformerConfig, self).__post_init__()
    if self.fp16 and self.bf16:
        raise ValueError(
            f'Only one of self.fp16: {self.fp16} and self.bf16 {self.bf16} should be True.'
        )
    args = get_args()
    world_size = args.tp_x if args.tp_2d else self.tensor_model_parallel_size
    if self.num_attention_heads % world_size != 0:
        if not args.unaligned_linear:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"tensor_model_parallel_size ({world_size})."
            )

    if self.ffn_hidden_size is None:
        self.ffn_hidden_size = 4 * self.hidden_size

    if self.kv_channels is None:
        self.kv_channels = self.hidden_size // self.num_attention_heads

    if self.num_query_groups is None:
        self.num_query_groups = self.num_attention_heads

    if self.num_query_groups % world_size != 0:
        if not args.unaligned_linear:
            raise ValueError(
                f"num_query_groups ({self.num_query_groups}) must be a multiple of "
                f"tensor_model_parallel_size ({world_size})."
            )

    if self.apply_query_key_layer_scaling:
        self.attention_softmax_in_fp32 = True

    if self.expert_model_parallel_size > 1 and self.num_moe_experts is None:
        raise ValueError(f'num_moe_experts must be non None to use expert-parallel.')

    if self.num_moe_experts is not None and self.num_moe_experts <= 0:
        raise ValueError(f'num_moe_experts must be non-negative.')

    if self.moe_expert_capacity_factor is not None:
        if self.moe_token_dispatcher_type != "alltoall":
            raise ValueError(
                f'moe_expert_capacity_factor only works with alltoall token dispatcher'
            )
        if self.moe_expert_capacity_factor < 0:
            self.moe_expert_capacity_factor = None
        if self.moe_router_load_balancing_type not in ["aux_loss", "none"]:
            raise ValueError(
                f'moe_expert_capacity_factor only works with aux_loss or none load balancing'
            )

    if self.moe_pad_expert_input_to_capacity:
        if self.moe_expert_capacity_factor is None:
            raise ValueError(
                f'moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity'
            )

    if self.cpu_offloading and (
        self.cpu_offloading_num_layers < 0 or self.cpu_offloading_num_layers >= self.num_layers
    ):
        raise ValueError(
            f'CPU offloading can be done only for layers less than {self.num_layers}'
        )

    if self.cpu_offloading and self.pipeline_model_parallel_size > 1:
        raise ValueError(
            f'Currently there is no support for Pipeline parallelism with CPU offloading'
        )

    if self.cpu_offloading and self.recompute_granularity is not None:
        raise ValueError(
            f'CPU offloading does not work when activation recomputation is enabled'
        )

    if self.recompute_granularity is not None:
        if self.recompute_granularity not in ['full', 'selective']:
            raise ValueError(
                f'When using recompute_granuarlity: {self.recompute_granularity} must be "full" or "selective".'
            )

        if self.recompute_method is not None:
            if self.recompute_method not in ['block', 'uniform']:
                raise ValueError(
                    f'recompute_method: {self.recompute_method} must be "block" or "uniform".'
                )
        elif self.recompute_granularity != 'selective':
            raise ValueError(
                f'Using recompute_granularity: {self.recompute_granularity} so recompute_method must be "block" or "uniform"'
            )

        if self.recompute_granularity != 'selective' and self.recompute_num_layers is None:
            raise ValueError(
                f'When using recompute_granularity: {self.recompute_granularity} recompute_num_layers must be between '
                f'1 and num_layers_per_pipeline_rank: {self.num_layers // self.pipeline_model_parallel_size}'
            )
        elif (
            self.recompute_granularity == 'selective' and self.recompute_num_layers is not None
        ):
            raise ValueError(
                f'When using recompute_granularity: {self.recompute_granularity} recompute_num_layers must be None.'
            )

        if self.distribute_saved_activations and self.sequence_parallel:
            raise ValueError(
                f'distribute_saved_activations: {self.distribute_saved_activations} must be false when sequence parallel is enabled: {self.sequence_parallel}'
            )

        if self.virtual_pipeline_model_parallel_size is not None:
            if not self.num_layers % self.virtual_pipeline_model_parallel_size == 0:
                raise ValueError(
                    f'num_layers: {self.num_layers} must be divisible by virtual_model_parallel_size {self.virtual_pipeline_model_parallel_size}'
                )

    if self.apply_query_key_layer_scaling:
        self.attention_softmax_in_fp32 = True

    if self.bias_activation_fusion:
        if self.activation_func not in [F.gelu, F.silu]:
            raise ValueError(
                "When bias_activation_fusion is True, activation function should be either gelu or swiglu"
            )
        if (
            self.activation_func == F.gelu
            and not self.gated_linear_unit
            and not self.add_bias_linear
        ):
            raise ValueError(
                "When bias_activation_fusion is True, gated_linear_unit is False, "
                "and activation function is gelu, add_bias_linear must also be True."
            )
    if self.activation_func_fp8_input_store:
        if self.activation_func != F.silu or not self.gated_linear_unit:
            raise ValueError("Storing activation input in FP8 is supported only for SwiGLU.")
    if self.apply_rope_fusion and self.rotary_interleaved:
        raise ValueError(f'rotary_interleaved does not work with apply_rope_fusion.')

    if self.init_method is None:
        self.init_method = init_method_normal(self.init_method_std)

    if self.output_layer_init_method is None:
        self.output_layer_init_method = scaled_init_method_normal(
            self.init_method_std, self.num_layers
        )

    if self.moe_extended_tp:
        if self.moe_token_dispatcher_type != 'allgather':
            raise ValueError(
                "Moe extended TP parallelism only applies to allgather based token dispatcher."
            )
        extended_tp_size = self.tensor_model_parallel_size * self.expert_model_parallel_size
        if self.ffn_hidden_size % extended_tp_size != 0:
            raise ValueError(
                f'ffn_hidden_size: {self.ffn_hidden_size} must be divisible by extended_tp_size {extended_tp_size}'
            )


def transformer_config_post_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self):
        fn(self)
        args = get_args()
        fields = []
        for key, value in vars(args).items():
            field_name = str(key)
            field_type = type(value)
            if not hasattr(self, key):
                field_def = (field_name, field_type, field(init=False))
                fields.append(field_def)
        self.__class__ = make_dataclass(self.__class__.__name__, fields=fields, bases=(self.__class__,))

        for key, value in vars(args).items():
            if not hasattr(self, key):
                setattr(self, key, value)
    return wrapper