# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""MindSpeed GatedDeltaNet subclass.

Inherits from Megatron-LM's ``GatedDeltaNet`` and overrides the three
FLA-dependent paths with MindSpeed's own operators:

* ``causal_conv1d``  → ``mindspeed.core.ssm.npu_causal_conv1d.causal_conv1d`` (F.conv1d)
* ``chunk_gated_delta_rule`` → ``mindspeed.core.ssm.npu_chunk_gated_delta_rule`` (Triton pipeline)
* ``l2norm``           → ``mindspeed.ops.triton.l2norm`` (Triton kernel + custom autograd)

All Megatron-main features — Context Parallel, THD packed sequences,
selective recompute — are inherited unchanged.
"""

import torch

from megatron.core.jit import jit_fuser
from megatron.core.ssm.gated_delta_net import GatedDeltaNet as _MegatronGatedDeltaNet

from mindspeed.core.ssm.npu_chunk_gated_delta_rule import chunk_gated_delta_rule as _ms_chunk_gated_delta_rule
from mindspeed.ops.triton.l2norm import l2norm as _ms_l2norm


class GatedDeltaNet(_MegatronGatedDeltaNet):
    """MindSpeed Gated Delta Net — inherits Megatron-main logic, overrides operators."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace chunk_gated_delta_rule with MindSpeed's Triton version.
        # Megatron's __init__ sets self.gated_delta_rule = chunk_gated_delta_rule
        # (module-level name from the Megatron file).  We overwrite it here.
        if not self.config.deterministic_mode:
            self.gated_delta_rule = _ms_chunk_gated_delta_rule

    @jit_fuser
    def _prepare_qkv_for_gated_delta_rule(self, qkv, gate, beta, alpha, batch, seq_len):
        """Override: use MindSpeed Triton l2norm instead of FLA l2norm."""
        query_key, value = torch.split(
            qkv,
            [2 * self.qk_dim_local_tp // self.cp_size, self.v_dim_local_tp // self.cp_size],
            dim=-1,
        )
        query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)

        if self.use_qk_l2norm:
            query_key = _ms_l2norm(query_key.contiguous())

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
