# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Any, Optional, Union
from functools import wraps
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
# pylint: disable=import-error

from megatron.core.transformer import ModuleSpec
from megatron.training import get_args
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer import TransformerConfig, build_module
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType


from mindspeed.core.transformer.multi_head_latent_attention.mla_utils import (
    yarn_get_mscale,
    apply_yarn_scaling,
)


@dataclass
class SelfAttentionSubmodules:
    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None
    linear_qb: Union[ModuleSpec, type] = None
    linear_kvb: Union[ModuleSpec, type] = None


def self_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        **attention_optional_kwargs: dict[str, Any]
    ):
        """
        Implement the Multi-Head Latent Attention.
        SelfAttention init func wrapper.

        Args:
            config (TransformerConfig): train config.
            submodules (SelfAttentionSubmodules): submodules.
            layer_number (int): layer number.
            attn_mask_type (AttnMaskType): attention mask type.
            attention_optional_kwargs (dict[str, Any]): kwargs.
        """
        fn(
            self, config, submodules, layer_number,
            attn_mask_type, **attention_optional_kwargs
        )

        # other feature's argument
        if hasattr(self.config, "use_flash_attn"):
            self.use_flash_attn = self.config.use_flash_attn
        else:
            self.use_flash_attn = False
        self.shape_order = self.config.shape_order
        self.qk_rope_head_dim = self.config.qk_rope_head_dim
        self.qk_nope_head_dim = self.config.qk_nope_head_dim
        self.q_lora_rank = self.config.q_lora_rank
        self.kv_lora_rank = self.config.kv_lora_rank
        self.v_head_dim = self.config.v_head_dim

        query_projection_size = (
            self.config.num_attention_heads * self.v_head_dim
        )
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        if self.q_lora_rank is None:
            self.q_rank = (
                self.config.num_attention_heads * self.q_head_dim
            )
            self.q_layernorm = None
        else:
            self.q_rank = self.q_lora_rank
            if submodules.q_layernorm is not None:
                self.q_layernorm = build_module(
                    submodules.q_layernorm,
                    hidden_size=self.q_lora_rank,
                    config=self.config,
                    eps=self.config.layernorm_epsilon,
                )
            else:
                self.q_layernorm = None

            self.linear_qb = build_module(
                submodules.linear_qb,
                self.q_lora_rank,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=(
                    self.config.add_bias_linear or
                    self.config.add_qkv_bias
                ),
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='qb',
            )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.q_rank + self.kv_lora_rank + self.qk_rope_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=(
                self.config.add_bias_linear or
                self.config.add_qkv_bias
            ),
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
        )

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.kv_lora_rank,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

        self.linear_kvb = build_module(
            submodules.linear_kvb,
            self.kv_lora_rank,
            self.config.num_attention_heads * (
                self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim
            ),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kvb',
        )

        self.linear_proj = build_module(
            submodules.linear_proj,
            query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )

    return wrapper


def attention_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
):
    """
    Implement the Multi-Head Latent Attention.
    SelfAttention derive from Attention, and doesn't override
    the forward func, and thus, need to patch Attention class.

    Args:
        hidden_states (torch.Tensor): input data tensor.
        attention_mask (torch.Tensor): attention mask tensor.
    """

    # the shape of hidden_states is [sq, b, h]

    """
    For self attention we just duplicate
    the rotary_pos_emb if it isn't already
    """
    if (
        rotary_pos_emb is not None and
        not isinstance(rotary_pos_emb, tuple)
    ):
        rotary_pos_emb = (rotary_pos_emb,) * 2

    q_len, bsz, _ = hidden_states.shape
    mixed_x_layer, _ = self.linear_qkv(hidden_states)

    # [sq, b, hp] --> [sq, b, ng, hn]
    q_a, compressed_kv, k_pe = torch.split(
        mixed_x_layer,
        [
            self.q_rank, self.kv_lora_rank,
            self.qk_rope_head_dim,
        ],
        dim=-1
    )

    if self.q_layernorm is None:
        q = q_a
    else:
        q, _ = self.linear_qb(self.q_layernorm(q_a))

    q = q.view(q_len, bsz, self.config.num_attention_heads, -1)

    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )

    k_pe = k_pe.view(q_len, bsz, 1, self.qk_rope_head_dim)
    kv, _ = self.linear_kvb(self.k_layernorm(compressed_kv))
    kv = kv.view(
        q_len,
        bsz,
        self.config.num_attention_heads,
        self.qk_nope_head_dim + self.v_head_dim
    )
    k_nope, value = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )

    if rotary_pos_emb is not None:
        q_pos_emb, k_pos_emb = rotary_pos_emb

        b, h, s, d = q_pe.shape
        q_pe = (
            q_pe.view(b, h, s, d // 2, 2)
                .transpose(4, 3)
                .reshape(b, h, s, d)
        )
        b, h, s, d = k_pe.shape
        k_pe = (
            k_pe.view(b, h, s, d // 2, 2)
                .transpose(4, 3)
                .reshape(b, h, s, d)
        )

        if packed_seq_params is not None:
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        q_pe = apply_rotary_pos_emb(
            q_pe,
            q_pos_emb,
            config=self.config,
            cu_seqlens=cu_seqlens_q
        )
        k_pe = apply_rotary_pos_emb(
            k_pe,
            k_pos_emb,
            config=self.config,
            cu_seqlens=cu_seqlens_kv
        )

    query = torch.cat([q_nope, q_pe], dim=-1)

    k_pe = k_pe.repeat(1, 1, query.shape[2], 1)
    key = torch.cat([k_nope, k_pe], dim=-1)

    if (
        self.use_flash_attn and
        self.q_head_dim != self.v_head_dim
    ):
        if self.shape_order == "BNSD":
            value = F.pad(value, [0, self.q_head_dim - self.v_head_dim])
        else:
            query = F.pad(query, [0, 256 - self.q_head_dim])
            key = F.pad(key, [0, 256 - self.q_head_dim])
            value = F.pad(value, [0, 256 - self.v_head_dim])

    # ==================================
    # core attention computation
    # ==================================
    attn_mask_type = AttnMaskType.causal
    if self.checkpoint_core_attention and self.training:
        core_attn_out = self._checkpointed_attention_forward(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
    else:
        core_attn_out = self.core_attention(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )

    if packed_seq_params is not None:
        # reshape to same output shape as unpacked case
        # (t, np, hn) -> (t, b=1, h=np*hn)
        # t is the pack size = sum (sq_i)
        # note that batch is a dummy dimension in the packed case
        core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

    if self.use_flash_attn:
        core_attn_out = core_attn_out.view(
            q_len,
            bsz,
            self.config.num_attention_heads,
            -1
        )
        core_attn_out = core_attn_out[:, :, :, : self.v_head_dim]
        core_attn_out = core_attn_out.reshape(
            q_len,
            bsz,
            self.config.num_attention_heads * self.v_head_dim
        )

    # =================
    # Output. [sq, b, h]
    # =================

    output, bias = self.linear_proj(core_attn_out)

    return output, bias


def dot_product_attention_init_wrapper(fn):
    """
    Add MultiHead Latent Attention related block
    """

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
            self, config,
            layer_number,
            attn_mask_type,
            attention_type,
            attention_dropout,
            softmax_scale,
            cp_comm_type
        )

        # add mla related block
        self.scale_mask_softmax.scale = True
        self.hidden_size_per_partition = (
            config.num_attention_heads * self.config.v_head_dim
        )
        self.q_head_dim = (
            self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        )

        self.softmax_scale = self.q_head_dim ** (-0.5)

        if self.config.rope_scaling_type is not None:
            mscale_all_dim = (
                self.config.rope_scaling_mscale_all_dim
                if self.config.rope_scaling_mscale_all_dim
                else 0
            )
            scaling_factor = self.config.rope_scaling_factor

            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.norm_factor = 1.0 / self.softmax_scale

    return wrapper


def get_gpt_layer_local_spec_wrapper(fn):
    @wraps(fn)
    def wrapper(
        # pylint: disable=unused-argument
        num_experts: Optional[int] = None,
        moe_grouped_gemm: Optional[bool] = False,
        qk_layernorm: Optional[bool] = False,
        multi_latent_attention: Optional[bool] = False,
        fp8: Optional[str] = None,
        moe_use_legacy_grouped_gemm: Optional[bool] = False
    ):
        res = fn(
            num_experts,
            moe_grouped_gemm,
            qk_layernorm,
            multi_latent_attention
        )

        res.submodules.self_attention.submodules = SelfAttentionSubmodules(
            linear_qkv=ColumnParallelLinear,
            core_attention=DotProductAttention,
            linear_proj=RowParallelLinear,
            q_layernorm=TENorm if qk_layernorm else IdentityOp,
            k_layernorm=TENorm if qk_layernorm else IdentityOp,
            linear_qb=ColumnParallelLinear,
            linear_kvb=ColumnParallelLinear
        )

        res.submodules.input_layernorm = TENorm
        res.submodules.pre_mlp_layernorm = TENorm
        return res

    return wrapper


def rotary_embedding_init_wrapper(fn):
    """
    Yet Another RoPE Extention.
    Megatron RotaryEmbedding.__init__ func wrapper.
    """

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        args = list(args)
        _args = get_args()
        if _args.rotary_base:
            if len(args) >= 5:
                args[4] = _args.rotary_base
            else:
                kwargs["rotary_base"] = _args.rotary_base

        fn(self, *args, **kwargs)

        if (
            hasattr(_args, "rope_scaling_type") and
            _args.rope_scaling_type == "yarn"
        ):
            self.inv_freq = apply_yarn_scaling(_args, self.inv_freq)

    return wrapper
