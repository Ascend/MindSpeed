# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from typing import Optional

import torch
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType

try:
    from einops import rearrange
except ImportError:
    rearrange = None


def _is_causal(attn_mask_type: Optional[AttnMaskType]) -> bool:
    return attn_mask_type == AttnMaskType.causal


def _get_attn_mask_type(self, attn_mask_type: Optional[AttnMaskType]) -> Optional[AttnMaskType]:
    return attn_mask_type if attn_mask_type is not None else self.attn_mask_type


def _validate_qkv_shape(query: Tensor, key: Tensor, value: Tensor, packed_seq_params):
    if packed_seq_params is not None:
        qkv_have_same_dim = key.dim() == query.dim() and value.dim() == query.dim()
        if query.dim() not in (3, 4) or not qkv_have_same_dim:
            raise AssertionError(
                f"flash-attn-npu varlen backend expects 3D or 4D q/k/v tensors, "
                f"got query={query.shape}, key={key.shape}, value={value.shape}."
            )
        qkv_have_dummy_batch = query.shape[1] == key.shape[1] == value.shape[1] == 1
        if query.dim() == 4 and not qkv_have_dummy_batch:
            raise AssertionError(
                "flash-attn-npu varlen backend only supports 4D packed q/k/v with dummy batch size 1, "
                f"got query={query.shape}, key={key.shape}, value={value.shape}."
            )
    else:
        qkv_are_4d = query.dim() == key.dim() == value.dim() == 4
        if not qkv_are_4d:
            raise AssertionError(
                f"flash-attn-npu dense backend expects 4D q/k/v tensors, "
                f"got query={query.shape}, key={key.shape}, value={value.shape}."
            )

    q_heads = query.shape[-2]
    kv_heads = key.shape[-2]
    if value.shape[-2] != kv_heads:
        raise AssertionError("flash-attn-npu backend requires key and value to have the same local KV head count.")
    if q_heads % kv_heads != 0:
        raise AssertionError("flash-attn-npu backend requires local query heads to be divisible by local KV heads.")
    if query.shape[-1] != key.shape[-1]:
        raise AssertionError("flash-attn-npu backend requires query and key to have the same head dimension.")


def _validate_flash_attn_npu_attention(
    self,
    query,
    key,
    value,
    attention_mask,
    attn_mask_type,
    packed_seq_params,
):
    effective_attn_mask_type = _get_attn_mask_type(self, attn_mask_type)

    if rearrange is None:
        raise ImportError("The flash-attn-npu backend requires einops.")

    if self.config.context_parallel_size != 1:
        raise AssertionError("flash-attn-npu backend does not support context parallelism.")

    if self.attention_dropout.p != 0:
        raise AssertionError("flash-attn-npu backend currently requires attention_dropout == 0.")

    if effective_attn_mask_type not in (AttnMaskType.causal, AttnMaskType.no_mask, None):
        raise AssertionError("flash-attn-npu backend currently only supports causal or no_mask attention.")

    if attention_mask is not None and not _is_causal(effective_attn_mask_type):
        raise AssertionError("flash-attn-npu backend does not support explicit non-causal attention masks.")

    if packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", None) != "thd":
        raise AssertionError("flash-attn-npu varlen backend requires packed_seq_params.qkv_format == 'thd'.")

    _validate_qkv_shape(query, key, value, packed_seq_params)


def _get_window_size(self):
    window_size = getattr(getattr(self, "scale_mask_softmax", None), "window_size", None)
    return window_size if window_size is not None else (-1, -1)


def _prepare_flash_attn_npu_cu_seqlens(cu_seqlens, total_tokens, name):
    if cu_seqlens.dim() != 1:
        raise AssertionError(f"flash-attn-npu varlen backend requires 1D {name}, got {cu_seqlens.shape}.")
    if cu_seqlens.numel() == 0:
        raise AssertionError(f"flash-attn-npu varlen backend requires non-empty {name}.")

    # MindSpeed reset-attention-mask stores actual sequence end positions
    # ([s0, s0+s1, ...]) while flash-attn varlen expects boundaries
    # ([0, s0, s0+s1, ...]).
    first = int(cu_seqlens[0].detach().cpu().item())
    if first != 0:
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])

    cu_cpu = cu_seqlens.detach().cpu()
    if int(cu_cpu[0]) != 0:
        raise AssertionError(f"flash-attn-npu varlen backend requires {name}[0] == 0.")
    if int(cu_cpu[-1]) != total_tokens:
        raise AssertionError(
            f"flash-attn-npu varlen backend requires {name}[-1] == total tokens, "
            f"got {int(cu_cpu[-1])} vs {total_tokens}."
        )

    seq_lens = cu_cpu[1:] - cu_cpu[:-1]
    if seq_lens.numel() == 0 or int(seq_lens.min()) <= 0:
        raise AssertionError(f"flash-attn-npu varlen backend requires positive sequence lengths in {name}.")

    return cu_seqlens.to(dtype=torch.int32).contiguous(), int(seq_lens.max())


def _get_flash_attn_npu_varlen_metadata(packed_seq_params, query_tokens, key_tokens):
    cache = getattr(packed_seq_params, "_flash_attn_npu_varlen_metadata", None)
    if cache is not None and cache.get("query_tokens") == query_tokens and cache.get("key_tokens") == key_tokens:
        return (
            cache["cu_seqlens_q"],
            cache["cu_seqlens_kv"],
            cache["max_seqlen_q"],
            cache["max_seqlen_kv"],
        )

    cu_seqlens_q, max_seqlen_q = _prepare_flash_attn_npu_cu_seqlens(
        packed_seq_params.cu_seqlens_q,
        query_tokens,
        "cu_seqlens_q",
    )
    cu_seqlens_kv, max_seqlen_kv = _prepare_flash_attn_npu_cu_seqlens(
        packed_seq_params.cu_seqlens_kv,
        key_tokens,
        "cu_seqlens_kv",
    )
    packed_seq_params._flash_attn_npu_varlen_metadata = {
        "query_tokens": query_tokens,
        "key_tokens": key_tokens,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_kv": cu_seqlens_kv,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_kv": max_seqlen_kv,
    }
    return cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv


def dot_product_attention_flash_attn_npu_forward(
    self,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor,
    attn_mask_type: AttnMaskType = None,
    attention_bias: Tensor = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
):
    """DotProductAttention.forward implementation backed by flash-attention-npu.

    flash-attention-npu follows Tri Dao's API shapes:
    - dense: [batch, seqlen, heads, head_dim]
    - varlen: [total_tokens, heads, head_dim]

    Megatron/MindSpeed dense attention reaches this function as [seqlen, batch, heads, head_dim],
    while packed sequence attention already uses the varlen [total_tokens, heads, head_dim] layout.
    """
    return _dot_product_attention_flash_attn_npu_forward_impl(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type,
        attention_bias,
        packed_seq_params,
    )


def _dot_product_attention_flash_attn_npu_forward_impl(
    self,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor,
    attn_mask_type: AttnMaskType = None,
    attention_bias: Tensor = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
):
    if attention_bias is not None:
        raise AssertionError("flash-attn-npu backend does not support attention_bias.")

    _validate_flash_attn_npu_attention(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type,
        packed_seq_params,
    )

    try:
        from flash_attn_npu import flash_attn_func, flash_attn_varlen_func
    except ImportError as exc:
        raise ImportError(
            "flash-attn-npu batch invariant backend requires flash-attention-npu. "
            "Install flash_attn_npu before using --use-flash-attn-npu-batch-invariant."
        ) from exc

    causal = _is_causal(_get_attn_mask_type(self, attn_mask_type))
    scale = self.softmax_scale
    window_size = _get_window_size(self)

    if packed_seq_params is not None:
        if packed_seq_params.cu_seqlens_q is None or packed_seq_params.cu_seqlens_kv is None:
            raise AssertionError("flash-attn-npu varlen backend requires packed_seq_params cu_seqlens.")
        if packed_seq_params.max_seqlen_q is None or packed_seq_params.max_seqlen_kv is None:
            raise AssertionError("flash-attn-npu varlen backend requires packed_seq_params max_seqlen.")

        # Megatron THD path may pass [T, 1, H, D] with a dummy batch dimension.
        has_dummy_batch_dim = query.dim() == 4
        if query.dim() == 4:
            query, key, value = [x.squeeze(1).contiguous() for x in (query, key, value)]
        else:
            query, key, value = [x.contiguous() for x in (query, key, value)]

        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = _get_flash_attn_npu_varlen_metadata(
            packed_seq_params,
            query.shape[0],
            key.shape[0],
        )
        output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            dropout_p=self.attention_dropout.p,
            softmax_scale=scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False,
            block_table=None,
        )
        if has_dummy_batch_dim:
            output = rearrange(output, "t h d -> t 1 (h d)")
        return output

    seq_length, batch_size = query.shape[0], query.shape[1]
    query, key, value = [rearrange(x, "s b h d -> b s h d").contiguous() for x in (query, key, value)]
    output = flash_attn_func(
        query,
        key,
        value,
        dropout_p=self.attention_dropout.p,
        softmax_scale=scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    )
    output = rearrange(output, "b s h d -> s b (h d)", s=seq_length, b=batch_size)
    return output
