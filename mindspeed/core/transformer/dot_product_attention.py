# coding=utf-8
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import math
from functools import wraps

import torch
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.utils import divide
from megatron.training import get_args
from megatron.core import parallel_state
from mindspeed.core.models.common.embeddings.rotary_pos_embedding import yarn_get_mscale
from mindspeed.model.alibi_mask import AlibiForFusionAttnSingleton


def dot_product_attention_init(
    self,
    config: TransformerConfig,
    layer_number: int,
    attn_mask_type: AttnMaskType,
    attention_type: str,
    attention_dropout: float = None,
    softmax_scale: float = None,
    cp_comm_type: str = None,
):
    cp_size = config.context_parallel_size
    config.context_parallel_size = 1

    super(DotProductAttention, self).__init__(config=config)
    assert self.config.context_parallel_size == 1, "Context parallelism is only supported by TEDotProductAttention!"

    assert self.config.window_size is None, "Sliding Window Attention is only supported by TEDotProductAttention!"

    self.layer_number = max(1, layer_number)
    self.attn_mask_type = attn_mask_type
    self.attention_type = attention_type  # unused for now

    projection_size = self.config.kv_channels * self.config.num_attention_heads
    args = get_args()
    # Per attention head and per partition values.
    world_size = parallel_state.get_tensor_model_parallel_world_size()
    self.hidden_size_per_partition = divide(projection_size, world_size)
    self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
    self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
    self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

    coeff = None
    if softmax_scale is None:
        self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
    else:
        self.softmax_scale = softmax_scale

    if self.config.apply_query_key_layer_scaling:
        coeff = self.layer_number
        self.softmax_scale /= coeff

    self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
    if self.config.apply_query_key_layer_scaling:
        coeff = self.layer_number
        self.norm_factor *= coeff

    self.scale_mask_softmax = FusedScaleMaskSoftmax(
        input_in_fp16=self.config.fp16,
        input_in_bf16=self.config.bf16,
        attn_mask_type=self.attn_mask_type,
        scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
        mask_func=attention_mask_func,
        softmax_in_fp32=self.config.attention_softmax_in_fp32,
        scale=coeff,
    )

    # Dropout. Note that for a single iteration, this layer will generate
    # different outputs on different number of parallel partitions but
    # on average it should not be partition dependent.
    self.attention_dropout = torch.nn.Dropout(
        self.config.attention_dropout if attention_dropout is None else attention_dropout
    )

    config.context_parallel_size = cp_size

    # add pse
    self.pse = None
    self.pse_type = args.alibi_fusion_attn_type

    if args.multi_head_latent_attention:
        self.scale_mask_softmax.scale = True
        self.hidden_size_per_partition = config.num_attention_heads * args.v_head_dim
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.softmax_scale = self.q_head_dim ** (-0.5)

        if args.rope_scaling_type is not None:
            mscale_all_dim = args.rope_scaling_mscale_all_dim if args.rope_scaling_mscale_all_dim else 0
            scaling_factor = args.yarn_scaling_factor

            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.norm_factor = 1.0 / self.softmax_scale

    if self.pse_type is None:
        self.pse_type = 1  # not use pse
    elif self.pse_type == 0:
        alibi = AlibiForFusionAttnSingleton.get_alibi_tensor_for_fusion_attn(
            args.seq_length, config.num_attention_heads, config.params_dtype, args.alibi_diagonal_opposite, 1024
        )
        self.pse = alibi
    elif self.pse_type in (2, 3):
        self.pse = AlibiForFusionAttnSingleton.get_alibi_slopes_for_fusion_attn(config.num_attention_heads)


def dot_product_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        if self.config.num_query_groups is None:
            self.config.num_query_groups = self.config.num_attention_heads
        self.num_attention_heads_per_partition = (
            self.config.num_attention_heads * self.num_query_groups_per_partition // self.config.num_query_groups
        )

    return wrapper
