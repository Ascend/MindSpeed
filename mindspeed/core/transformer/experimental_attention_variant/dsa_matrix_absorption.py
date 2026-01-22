# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025; Huawei Technologies Co., Ltd.  All rights reserved.

from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch
import torch_npu

try:
    from einops import rearrange

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

from megatron.core.models.backends import BackendSpecProvider
from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    fine_grained_offloading_group_commit,
    fine_grained_offloading_group_start,
    get_fine_grained_offloading_context,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import deprecate_inference_params, get_pg_size
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.multi_latent_attention import MultiLatentAttention


try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELinear
    )
    from megatron.core.post_training.modelopt.layers import Linear

    HAVE_TE = True
except ImportError:
    TEColumnParallelLinear, TELinear, Linear, set_save_original_input = None, None, None, None
    HAVE_TE = False


def compute_dsa_indexer_loss(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection,
) -> torch.Tensor:
    """
    Extended to support a mismatched number of key and query heads.
    This enables functionality such as cross-attention with matrix absorption.
    """
    sq, b, np, hn = query.size()
    skv, b, nkv, hn = key.size()
    sk = key.size(0)

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key = key.permute(1, 2, 3, 0).reshape(b * nkv, hn, sk)
    # Compute attention scores [b * np, sq, sk]
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    # causal_mask [sq, sk]
    causal_mask = torch.triu(
        torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)

    # [b, np, sq, skv] + [1, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores += causal_mask.view(1, 1, sq, sk)
    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores += index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores += index_mask

    # [b, np, sq, sk] -> [b, np, sq, sk]
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)
    # [b, sq, sk] -> [b, sq, sk]
    index_scores = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)

    # Sum attention scores across heads.
    # [batch, heads, seqlen_q, seqlen_k] -> [batch, seqlen_q, seqlen_k]
    attention_scores = attention_scores.sum(dim=1)
    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores.contiguous(), group=pg_collection.tp)
    # L1 normalize target on the last dimension. Doesn't use abs() because attention_scores are
    # obtained from softmax so they are already non-negative.
    attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)

    # Compute KL divergence: KL(target || index) = target(x) * log(target(x) / index(x))
    # kl_per_element [b, sq, sk]
    kl_per_element = attention_scores * (
        torch.log(attention_scores + 1e-10) - torch.log(index_scores + 1e-10)
    )

    # [b, sq, sk] -> [b, sq] -> [1]
    # Each token has same weight in the loss.
    kl_div = kl_per_element.sum(dim=-1).mean()

    # Scale by coefficient.
    indexer_loss = kl_div * loss_coeff

    return indexer_loss


def unfused_dsa_fn(query, key, value, topk_indices, softmax_scale):
    """
    Extended to support a mismatched number of key and query heads.
    This enables functionality such as cross-attention with matrix absorption.
    """
    sq, b, np, hn = query.size()
    skv, b, nkv, hn = key.size()
    skv = key.size(0)
    hnv = value.size(3)
    # ===================================
    # Raw attention scores [b, np, sq, skv]
    # ===================================
    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [skv, b, np, hn] -> [b, np, hn, skv] -> [b * np, hn, skv]
    key = key.permute(1, 2, 3, 0).reshape(b * nkv, hn, skv)
    # Compute attention scores [b * np, sq, skv]
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    # Reshape to [b, np, sq, skv]
    attention_scores = attention_scores.reshape(b, np, sq, skv)

    # ===================================
    # Apply sparse mask from indexer
    # ===================================
    index_mask = torch.full((b, sq, skv), float("-inf"), device=attention_scores.device)
    index_mask.scatter_(-1, topk_indices, 0)
    causal_mask = torch.triu(
        torch.full((sq, skv), float('-inf'), dtype=torch.float32, device=index_mask.device),
        diagonal=1,
    )
    # [b, sq, skv] + [1, sq, skv] -> [b, sq, skv]
    index_mask += causal_mask.view(1, sq, skv)
    # [b, np, sq, skv] + [b, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores += index_mask.unsqueeze(1)
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)

    # ===================================
    # Output
    # ===================================
    # [skv, b, np, hnv] -> [b, np, skv, hnv] -> [b * np, skv, hnv]
    value = value.permute(1, 2, 0, 3).reshape(b * nkv, skv, hnv)
    # Reshape attention_scores: [b, np, sq, skv] -> [b * np, sq, skv]
    attention_scores = attention_scores.reshape(b * np, sq, skv)
    # Compute output: [b * np, sq, hnv]
    output = torch.bmm(attention_scores.to(value.dtype), value)
    # Reshape output: [b * np, sq, hnv] -> [b, np, sq, hnv] -> [sq, b, np, hnv]
    output = output.reshape(b, np, sq, hnv).permute(2, 0, 1, 3).contiguous()
    # Flatten: [sq, b, np, hnv] -> [sq, b, np * hnv]
    output = output.reshape(sq, b, np * hnv)
    return output


@dataclass
class MLASelfAttentionAbsorbSubmodules:
    """Submodules for the MLA self-attention layer."""

    linear_q_proj: Union[ModuleSpec, type] = None
    linear_q_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_down_proj: Union[ModuleSpec, type] = None
    linear_k_up_proj: Union[ModuleSpec, type] = None
    linear_v_up_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    kv_layernorm: Union[ModuleSpec, type] = None


def get_dsa_module_spec_for_backend(
    backend: BackendSpecProvider,
    qk_layernorm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    mla_down_proj_use_column_parallel: Optional[bool] = False,
    normalization: Optional[str] = None,
    fallback_to_eager_attn: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for Sparse Attention."""
    if not multi_latent_attention:
        RuntimeError("Currently only MLA supports sparse attention.")
    if qk_l2_norm is True:
        RuntimeError("qk_l2_norm is not supported with MLA.")
    if fallback_to_eager_attn is True:
        RuntimeError("Fallback to eager attention is not supported with DSA.")

    linear_q_down_proj = (
        backend.column_parallel_linear() if mla_down_proj_use_column_parallel else backend.linear()
    )
    linear_kv_down_proj = (
        backend.column_parallel_linear() if mla_down_proj_use_column_parallel else backend.linear()
    )
    linear_q_up_proj = backend.column_parallel_linear()
    linear_k_up_proj = backend.column_parallel_linear()
    linear_v_up_proj = backend.column_parallel_linear()

    # Because TransformerEngine does not support sparse attention yet, we use local
    # implementation whether the backend is TransformerEngine or not.
    core_attention = ModuleSpec(
        module=DSAttention,
        submodules=DSAttentionSubmodules(
            indexer=ModuleSpec(
                module=DSAIndexer,
                submodules=DSAIndexerSubmodules(
                    linear_wq_b=backend.linear(),
                    linear_wk=backend.linear(),
                    k_norm=backend.layer_norm(rms_norm=False, for_qk=True),
                    linear_weights_proj=backend.linear(),
                ),
            )
        ),
    )

    # Adjust for RMS norm.
    rms_norm = normalization == "RMSNorm"
    qk_norm = backend.layer_norm(rms_norm=rms_norm, for_qk=True) if qk_layernorm else IdentityOp

    attention = ModuleSpec(
        module=MLASelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=MLASelfAttentionAbsorbSubmodules(
            linear_q_proj=backend.column_parallel_linear(),
            linear_q_down_proj=linear_q_down_proj,
            linear_q_up_proj=linear_q_up_proj,
            linear_kv_down_proj=linear_kv_down_proj,
            linear_k_up_proj=linear_k_up_proj,
            linear_v_up_proj=linear_v_up_proj,
            core_attention=core_attention,
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=qk_norm,
            kv_layernorm=qk_norm,
        ),
        metainfo={"fuse_input_layernorm": False},
    )

    return attention


class MLASelfAttentionAbsorb(MLASelfAttention):
    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: ProcessGroupCollection = None,
    ) -> None:

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        MultiLatentAttention.__init__(
            self,
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        if self.config.q_lora_rank is None:
            # Not projecting query
            self.linear_q_proj = build_module(
                submodules.linear_q_proj,
                self.config.hidden_size,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_proj',
            )

        else:
            q_down_proj_kwargs = {}
            if submodules.linear_q_down_proj in [TELinear]:
                q_down_proj_kwargs['parallel_mode'] = 'duplicated'
            elif submodules.linear_q_down_proj in [
                Linear,
                TEColumnParallelLinear,
                ColumnParallelLinear,
            ]:
                q_down_proj_kwargs['gather_output'] = False
            else:
                raise ValueError(f"Unsupported linear_q_down_proj: {submodules.linear_q_down_proj}")

            self.linear_q_down_proj = build_module(
                submodules.linear_q_down_proj,
                self.config.hidden_size,
                self.config.q_lora_rank,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_down_proj',
                skip_weight_param_allocation=False,
                tp_group=(
                    pg_collection.tp
                    if q_down_proj_kwargs.get('parallel_mode') != 'duplicated'
                    else None
                ),
                **q_down_proj_kwargs,
            )

            self.linear_q_up_proj = build_module(
                submodules.linear_q_up_proj,
                self.config.q_lora_rank,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_up_proj',
                tp_group=pg_collection.tp,
            )

        kv_down_proj_kwargs = {}
        if submodules.linear_kv_down_proj in [TELinear]:
            kv_down_proj_kwargs['parallel_mode'] = 'duplicated'
        elif submodules.linear_kv_down_proj in [
            Linear,
            TEColumnParallelLinear,
            ColumnParallelLinear,
        ]:
            kv_down_proj_kwargs['gather_output'] = False
        else:
            raise ValueError(f"Unsupported linear_kv_down_proj: {submodules.linear_kv_down_proj}")

        self.linear_kv_down_proj = build_module(
            submodules.linear_kv_down_proj,
            self.config.hidden_size,
            self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_down_proj',
            skip_weight_param_allocation=False,
            tp_group=(
                pg_collection.tp
                if kv_down_proj_kwargs.get('parallel_mode') != 'duplicated'
                else None
            ),
            **kv_down_proj_kwargs,
        )

        self.linear_k_up_proj = build_module(
            submodules.linear_k_up_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads * self.config.qk_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='k_up_proj',
            tp_group=pg_collection.tp,
        )

        self.linear_v_up_proj = build_module(
            submodules.linear_v_up_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads * self.config.v_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='v_up_proj',
            tp_group=pg_collection.tp,
        )

        if self.config.q_lora_rank is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.config.q_lora_rank,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )

        self.kv_layernorm = build_module(
            submodules.kv_layernorm,
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):

        """Forward pass for multi-latent attention"""
        # the shape of hidden_states is [sq, b, h]

        if rotary_pos_emb is not None:
            RuntimeError("Rotary position embeddings should not be passed into MLA.")
        if attention_bias is not None:
            RuntimeError("Attention bias should not be passed into MLA.")
        if not (rotary_pos_cos is None and rotary_pos_sin is None):
            RuntimeError("MLA does not support Flash Decoding")
        if rotary_pos_cos_sin:
            RuntimeError("Flash-infer rope has not been tested with MLA.")
        if self.training and self.cache_mla_latents:
            RuntimeError("cache_mla_latents conflicts with training.")

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]
        if self.config.experimental_attention_variant is None:
            query, key, value = self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                packed_seq_params,
                inference_context=inference_context,
            )
        elif self.config.experimental_attention_variant == "dsa":
            query, key, value, q_compressed, _ = self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                packed_seq_params,
                inference_context=inference_context,
                return_compressed_tensors=True,
            )
        else:
            raise ValueError(
                f"Unsupported experimental attention variant: "
                f"{self.config.experimental_attention_variant}"
            )

        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        query, key, value, _, attn_mask_type, block_table = self._adjust_key_value_for_inference(
            inference_context, query, key, value, rotary_pos_emb=None
        )

        query = query.contiguous()
        key = key.contiguous()

        # Value is none during decode for absorption
        if value is not None:
            value = value.contiguous()

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            if self.offload_core_attention and self.training:
                query = fine_grained_offloading_group_start(query, name="core_attn")

            if inference_context is None or inference_context.is_static_batching():
                with get_fine_grained_offloading_context(self.offload_core_attention):
                    if self.config.experimental_attention_variant is None:
                        core_attn_out = self.core_attention(
                            query,
                            key,
                            value,
                            attention_mask,
                            packed_seq_params=packed_seq_params,
                            attn_mask_type=attn_mask_type,
                        )
                    elif self.config.experimental_attention_variant == "dsa":
                        # For dsa we need to pass in the original hidden states and the compressed
                        # query representation.
                        core_attn_out = self.core_attention(
                            query,
                            key,
                            value,
                            x=hidden_states,
                            qr=q_compressed,
                            attention_mask=attention_mask,
                            attn_mask_type=attn_mask_type,
                            attention_bias=None,
                            packed_seq_params=packed_seq_params,
                        )
                    else:
                        raise ValueError(
                            f"Unsupported attention variant: "
                            f"{self.config.experimental_attention_variant}"
                        )
            elif self.cache_mla_latents:
                # Dynamic batching attention kernel.
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    block_table,
                )
                # Only rearrange if not in absorption mode (Flash MLA handles format correctly)
                if not inference_context.is_decode_only():
                    core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')
            if self.offload_core_attention and self.training:
                (core_attn_out,) = fine_grained_offloading_group_commit(
                    core_attn_out, name="core_attn", forced_released_tensors=[query, key, value]
                )

        # Multiply by up v if absorbing
        core_attn_out = core_attn_out.view(core_attn_out.size(0), core_attn_out.size(1), self.num_attention_heads_per_partition, -1)
        v_weight = self.linear_v_up_proj.weight
        W_UV = v_weight.view(self.num_attention_heads_per_partition, self.config.v_head_dim, -1)
        W_UV_T = W_UV.permute(0, 2, 1).contiguous()
        core_attn_out = torch.einsum("sbhc,hcv->sbhv", core_attn_out, W_UV_T)
        core_attn_out = core_attn_out.contiguous()

        # Flatten back: [seq, batch, num_heads * v_head_dim]
        core_attn_out = core_attn_out.view(core_attn_out.size(0), core_attn_out.size(1), -1)

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            if self.qkv_up_checkpoint is None:
                RuntimeError("qkv_up_checkpoint is None in mla.")
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # =================
        # Output. [sq, b, h]
        # =================
        if self.offload_attn_proj:
            core_attn_out = fine_grained_offloading_group_start(core_attn_out, name="attn_proj")
        with get_fine_grained_offloading_context(self.offload_attn_proj):
            output, bias = self.linear_proj(core_attn_out)
        if self.offload_attn_proj:
            output, bias = fine_grained_offloading_group_commit(
                output, bias, name="attn_proj", forced_released_tensors=[core_attn_out]
            )

        return output, bias

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
        return_compressed_tensors=False,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # s = sequence length, b = batch size, h = hidden size, n = num attention heads
        # Attention heads [s, b, n*h]
        if not hidden_states.ndim == 3:
            RuntimeError(f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D")
        if packed_seq_params is not None:
            if not (packed_seq_params.local_cp_size is None):
                RuntimeError("hybrid_context_parallel is not supported with MLA yet and is planned for future. \
                Please disable hybrid_context_parallel.")

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, None, hidden_states, self.config, packed_seq_params
        )

        # rotary_pos_emb:[s, b, 1, 64]
        mscale = 1.0
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len, packed_seq_params=packed_seq_params
            )
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(
                rotary_seq_len, packed_seq_params=packed_seq_params
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        if self.config.q_lora_rank is not None:
            # if linear_q_down_proj is ColumnParallelLinear:
            #     q_compressed: [s, b, q_lora_rank / TP]
            # elif linear_q_down_proj is Linear:
            #     q_compressed: [s / TP, b, q_lora_rank]
            q_compressed, _ = self.linear_q_down_proj(hidden_states)

            # When output is sharded (ColumnParallelLinear), two things are needed to be
            # identical to a normal Linear.
            #   1. Manually gather output to restore output dim q_lora_rank;
            #   2. Scatter sequence back to s / TP if sequence-parallel since it was
            #      gathered by ColumnParallelLinear.
            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(q_compressed)
        else:
            q_compressed = hidden_states

        # if linear_kv_down_proj is ColumnParallelLinear:
        #     kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim) / TP]
        # elif linear_kv_down_proj is Linear:
        #     kv_combined: [s / TP, b, (kv_lora_rank + qk_pos_emb_head_dim)]
        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        if kv_combined.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
            # kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim)]
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            # kv_compressed:[s, b, kv_lora_rank], k_pos_emb: [s, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
        else:
            # kv_compressed:[s / TP, b, kv_lora_rank], k_pos_emb: [s / TP, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if get_pg_size(self.tp_group) > 1 and self.config.sequence_parallel:
                # k_pos_emb: [s, b, qk_pos_emb_head_dim]
                kv_compressed = gather_from_sequence_parallel_region(kv_compressed, group=self.tp_group)
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb, group=self.tp_group)

        if packed_seq_params is not None:
            # If sequence packing, TE expect [t, h, d] shaped qkv input.
            # In Megatron-Core, the qkv shape is [t, 1, h, d].
            # So we need to reshape qkv from [t, 1, h, d] to [t, h, d].
            q_compressed = q_compressed.squeeze(1)
            kv_compressed = kv_compressed.squeeze(1)
            k_pos_emb = k_pos_emb.squeeze(1)

        # =========================================
        # Apply norm
        # =========================================

        if self.config.q_lora_rank is not None:
            # q_compressed: [num_tokens, q_lora_rank]
            q_compressed = self.q_layernorm(q_compressed)

        kv_compressed = self.kv_layernorm(kv_compressed)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================

        def qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb):
            """
            Apply the up projection and RoPE to the query and key.
            When sequence packing enabled, the input tensors adopt a packed shape of [t, ...];
            otherwise, they maintain the unpacked shape [s, b, ...]. In subsequent code comments,
            we uniformly use [num_tokens, ...] to denote [s, b, ...] or [t, ...] for two cases.
            """
            if self.config.q_lora_rank is not None:
                # q_compressed: [num_tokens, q_lora_rank]
                # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
                q, _ = self.linear_q_up_proj(q_compressed)
            else:
                # q_compressed: [num_tokens, hidden_size]
                # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
                q, _ = self.linear_q_proj(q_compressed)

            # q: [num_tokens, n, q_head_dim]
            q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)

            k_lat = torch.unsqueeze(kv_compressed, -2)
            v_lat = torch.unsqueeze(kv_compressed, -2)
            k_no_pe = k_lat
            value = v_lat

            # [num_tokens, qk_pos_emb_head_dim] -> [num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            q_len = q.size()[0]
            if inference_context is not None:
                # add offset to the sequence start for inference
                sequence_start = inference_context.sequence_len_offset
                sequence_end = sequence_start + q_len
                rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]
            elif packed_seq_params is None or self.config.context_parallel_size == 1:
                # Shorten rotary_pos_emb to the sequence length when inference_params
                # is not provided. This makes sure we can run forward directly with
                # any sequence length. During training, the sequence length is always
                # the full rotary_pos_emb length, except for sequence packing + CP.
                # When sequence packing and context parallel are both enabled, the
                # position embedding will not split rotary_pos_emb, so it may exceed
                # the sequence length on this CP rank, but we need the full rotary_pos_emb
                # to cover the full sequence, so we do not shorten it here.
                rotary_pos_emb = rotary_pos_emb[0:q_len]

            # q_no_pe: [num_tokens, n, qk_head_dim]
            # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
            q_no_pe, q_pos_emb = torch.split(
                q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
            )

            # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
            q_pos_emb = apply_rotary_pos_emb(
                q_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
                mscale=mscale,
                cp_group=self.pg_collection.cp,
            )
            # k_pos_emb:[num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = apply_rotary_pos_emb(
                k_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
                mscale=mscale,
                cp_group=self.pg_collection.cp,
            )

            # Compute query components. Multiply by up k if absorbing
            kv_nope_weight = self.linear_k_up_proj.weight
            W_UK = kv_nope_weight.view(self.num_attention_heads_per_partition, self.config.qk_head_dim, -1)
            q_no_pe = torch.einsum("sbhd,hdk->sbhk", q_no_pe, W_UK)
            # query: [num_tokens, n, (qk_head_dim + v_head_dim)]
            query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

            # key: [num_tokens, n, (qk_head_dim + v_head_dim)]
            if k_pos_emb.ndim == 4:
                k_pos_emb = k_pos_emb.expand(k_pos_emb.shape[0], k_pos_emb.shape[1], 1, k_pos_emb.shape[3])
            else:
                if not k_pos_emb.ndim == 3:
                    RuntimeError("the dim of k_pos_emb should be 3")
                k_pos_emb = k_pos_emb.expand(k_pos_emb.shape[0], 1, k_pos_emb.shape[2])
            key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            return query, key, value

        if self.recompute_up_proj:
            quantization = self.config.fp8 or self.config.fp4
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=quantization)
            query, key, value = self.qkv_up_checkpoint.checkpoint(
                qkv_up_proj_and_rope_apply, q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )
        else:
            query, key, value = qkv_up_proj_and_rope_apply(
                q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )

        if return_compressed_tensors:
            return query, key, value, q_compressed, kv_compressed
        else:
            return query, key, value