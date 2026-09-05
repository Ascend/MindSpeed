# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""MindSpeed-owned DeepSeek-V4 modules used by deepseek_v4_cp_algo."""

from .attention import (
    DeepSeekV4MTPSelfAttentionCP,
    DeepSeekV4SelfAttentionCP,
    DeepSeekV4SelfAttentionCPSubmodules,
    get_deepseek_v4_cp_self_attn_submodules,
)
from .compressor import Compressor, CompressorSubmodules, get_compressor_spec
from .indexer import (
    DSAIndexer,
    DSAIndexerSubmodules,
    get_dsa_indexer_spec,
)
from .indexer_loss import (
    DSAIndexerLossLoggingHelper,
    set_deepseek_v4_cp_indexer_loss_scale,
    track_deepseek_v4_cp_indexer_metrics,
)

__all__ = [
    "Compressor",
    "CompressorSubmodules",
    "DSAIndexer",
    "DSAIndexerLossLoggingHelper",
    "DSAIndexerSubmodules",
    "DeepSeekV4MTPSelfAttentionCP",
    "DeepSeekV4SelfAttentionCP",
    "DeepSeekV4SelfAttentionCPSubmodules",
    "get_compressor_spec",
    "get_deepseek_v4_cp_self_attn_submodules",
    "get_dsa_indexer_spec",
    "set_deepseek_v4_cp_indexer_loss_scale",
    "track_deepseek_v4_cp_indexer_metrics",
]
