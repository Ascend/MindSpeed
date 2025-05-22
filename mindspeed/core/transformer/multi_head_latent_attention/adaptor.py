# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from typing import Union
from functools import wraps

from megatron.core import parallel_state
from megatron.core.models.common.embeddings import (
    YarnRotaryEmbedding,
    _yarn_get_mscale,
)
from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.multi_latent_attention import MLASelfAttentionSubmodules
from megatron.core.utils import divide


def multi_latent_attention_init_impl(
    self,
    config: MLATransformerConfig,
    submodules: Union[MLASelfAttentionSubmodules],
    layer_number: int,
    attn_mask_type: AttnMaskType,
    attention_type: str,
    cp_comm_type: str = None,
) -> None:
    world_size = parallel_state.get_tensor_model_parallel_world_size()
    assert (
        world_size == 1
    ), "MLA is not supported with Tensor Parallelism yet, \
    use Expert Parallelism and Pipeline Parallelism for better performance."

    Attention.__init__(
        self,
        config=config,
        submodules=submodules,
        layer_number=layer_number,
        attention_type=attention_type,
        attn_mask_type=attn_mask_type,
    )

    self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads

    self.q_head_dim = self.config.qk_head_dim + self.config.qk_pos_emb_head_dim

    # Overwrite the base class kv shape to support MLA inference
    self.key_hidden_size = self.q_head_dim
    self.val_hidden_size = self.config.v_head_dim

    self.recompute_up_proj = (
        self.config.recompute_granularity == 'selective'
        and "mla_up_proj" in self.config.recompute_modules
    )
    self.qkv_up_checkpoint = None

    mscale = _yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale)
    self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)

    if self.config.rope_type == "rope":
        self.rotary_pos_emb = RotaryEmbedding(
            self.config.qk_pos_emb_head_dim,
            rotary_percent=self.config.rotary_percent,
            rotary_base=self.config.rotary_base,
        )
    elif self.config.rope_type == "yarn":
        assert not self.config.apply_rope_fusion, "MLA Yarn RoPE does not support RoPE fusion"
        self.rotary_pos_emb = YarnRotaryEmbedding(
            self.config.qk_pos_emb_head_dim,
            rotary_base=self.config.rotary_base,
            scaling_factor=self.config.rotary_scaling_factor,
            original_max_position_embeddings=self.config.max_position_embeddings,
            beta_fast=self.config.beta_fast,
            beta_slow=self.config.beta_slow,
            mscale=self.config.mscale,
            mscale_all_dim=self.config.mscale_all_dim,
        )
    else:
        raise ValueError(
            f"Unsupported RoPE type: {self.config.rope_type}, supported types are "
            "'rope' and 'yarn'"
        )



    # Megatron use TEDotProductAttention
    # we use DotProductAttention
    self.core_attention = build_module(
        submodules.core_attention,
        config=self.config,
        layer_number=self.layer_number,
        attn_mask_type=self.attn_mask_type,
        attention_type=self.attention_type,
        softmax_scale=self.softmax_scale,
        cp_comm_type=cp_comm_type,
    )

    # Output.
    self.linear_proj = build_module(
        submodules.linear_proj,
        self.query_projection_size,
        self.config.hidden_size,
        config=self.config,
        init_method=self.config.output_layer_init_method,
        bias=self.config.add_bias_linear,
        input_is_parallel=True,
        skip_bias_add=True,
        is_expert=False,
        tp_comm_buffer_name='proj',
    )


def dot_product_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        cp_comm_type: str = None
    ):

        fn(
            self,
            config,
            layer_number,
            attn_mask_type,
            attention_type,
            attention_dropout=attention_dropout,
            softmax_scale=softmax_scale,
            cp_comm_type=cp_comm_type,
        )

        projection_size = self.config.v_head_dim * self.config.num_attention_heads

        self.hidden_size_per_partition = self.config.v_head_dim * self.config.num_attention_heads
        self.hidden_size_per_partition_head = divide(projection_size, config.num_attention_heads)

    return wrapper
