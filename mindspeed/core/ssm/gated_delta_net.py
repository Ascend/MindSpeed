# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""MindSpeed GatedDeltaNet subclass with a lazily selected backend.

All Megatron-main features — Context Parallel, THD packed sequences,
selective recompute — are inherited unchanged.
"""

from functools import wraps

import torch

from megatron.core.jit import jit_fuser
import megatron.core.ssm.gated_delta_net as megatron_gdn
from megatron.core.ssm.gated_delta_net import GatedDeltaNet as _MegatronGatedDeltaNet

from mindspeed.core.ssm.gdn_backend import load_gdn_operators


def gdn_transformer_config_post_init_wrapper(fn):
    """Bypass the variable-sequence MoE check for dense GDN configs only."""

    @wraps(fn)
    def wrapper(self):
        is_dense_variable_length_gdn = (
            getattr(self, "experimental_attention_variant", None) == "gated_delta_net"
            and getattr(self, "num_moe_experts", None) is None
            and getattr(self, "moe_token_dispatcher_type", None) == "allgather"
            and getattr(self, "variable_seq_lengths", False)
        )
        if not is_dense_variable_length_gdn:
            return fn(self)

        original_variable_seq_lengths = self.variable_seq_lengths
        self.variable_seq_lengths = False
        try:
            return fn(self)
        finally:
            self.variable_seq_lengths = original_variable_seq_lengths

    return wrapper


class GatedDeltaNet(_MegatronGatedDeltaNet):
    """MindSpeed Gated Delta Net — inherits Megatron-main logic, overrides operators."""

    def __init__(self, *args, cp_comm_type=None, **kwargs):
        operators = load_gdn_operators()
        megatron_gdn.HAVE_FLA = True
        megatron_gdn.causal_conv1d = operators.causal_conv1d
        super().__init__(*args, **kwargs)
        self._gdn_l2norm = operators.l2norm
        if not self.config.deterministic_mode:
            self.gated_delta_rule = operators.chunk_gated_delta_rule

    def _resolve_cu_seqlens(self, cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name, cp_size: int = 1):
        """Normalize MindSpeed endpoint metadata to Megatron's cu_seqlens contract."""
        cu_seqlens = cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens_actual
        cu_seqlens = cu_seqlens.reshape(-1)

        total_cu = cu_seqlens[-1].cpu().item()
        if cu_seqlens.numel() > 0 and int(cu_seqlens[0].item()) != 0 and total_cu == total_seq_len:
            cu_seqlens = torch.cat(
                [
                    torch.zeros(1, dtype=cu_seqlens.dtype, device=cu_seqlens.device),
                    cu_seqlens,
                ]
            )
        if total_cu != total_seq_len:
            raise ValueError(
                f"GDN: {name}[-1]={total_cu} does not match "
                f"total_sequence_length={total_seq_len}. "
                f"({cu_seqlens_padded=}, {cu_seqlens_actual=})."
            )

        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if (seq_lengths % cp_size != 0).any():
            raise ValueError(
                f"All per-sequence lengths in cu_seqlens must be divisible by cp_size={cp_size}, "
                f"but got lengths: {seq_lengths.tolist()}"
            )

        return cu_seqlens

    @jit_fuser
    def _prepare_qkv_for_gated_delta_rule(self, qkv, gate, beta, alpha, batch, seq_len):
        """Override: use the selected backend's l2norm operator instead of FLA's."""
        query_key, value = torch.split(
            qkv,
            [2 * self.qk_dim_local_tp // self.cp_size, self.v_dim_local_tp // self.cp_size],
            dim=-1,
        )
        query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)

        if self.use_qk_l2norm:
            query_key = self._gdn_l2norm(query_key.contiguous())

        split_size = self.qk_dim_local_tp // self.key_head_dim // self.cp_size
        query, key = torch.split(query_key, [split_size, split_size], dim=2)

        if self.num_value_heads // self.num_key_heads > 1:
            repeat_factor = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous()
        alpha = alpha.contiguous()

        return query, key, value, gate, beta, alpha
