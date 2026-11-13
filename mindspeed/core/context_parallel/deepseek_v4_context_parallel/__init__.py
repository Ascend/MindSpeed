# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DeepSeek-V4 Context Parallel support.

This package provides context parallelism (CP) support for DeepSeek-V4 style
sparse attention patterns, including:
- Compressed KV preparation and communication across CP ranks
- SMLA (Sparse Multi-Level Attention) input construction
- Window exchange for P2P-based CP communication
"""

# Data types
from ._types import (
    DeepSeekV4CPAlignmentDescriptor,
    DeepSeekV4CPCompressedKV,
    DeepSeekV4CPCompressionCandidates,
    DeepSeekV4CPCompressContext,
    DeepSeekV4CPMetadata,
    DeepSeekV4CPPendingCompressedKV,
    DeepSeekV4CPPackedSeqMetadata,
    DeepSeekV4CPRuntimeMetadata,
    DeepSeekV4CPSMLAInputs,
)

# Distributed communication
from ._distributed import (
    exchange_deepseek_v4_packed_previous_window,
    exchange_deepseek_v4_previous_tails,
    exchange_deepseek_v4_previous_window,
)

# Compressed KV operations
from ._compressed_kv import (
    compact_deepseek_v4_compressed_kv,
    launch_deepseek_v4_allgather_compressed_kv,
    prepare_deepseek_v4_compression_candidates_for_cp,
    wait_deepseek_v4_compressed_kv,
)

# SMLA input construction
from ._smla_inputs import (
    align_deepseek_v4_cp_tensor,
    build_deepseek_v4_causal_cmp_sparse_indices,
    build_deepseek_v4_cmp_cu_seqlens,
    build_deepseek_v4_cmp_residual_kv,
    build_deepseek_v4_cp_packed_seq_metadata,
    build_deepseek_v4_cp_smla_inputs,
    build_deepseek_v4_owned_runtime_metadata,
    flatten_deepseek_v4_cp_tensor_to_tnd,
    remap_deepseek_v4_cp_sparse_indices,
    validate_deepseek_v4_c4a_cmp_sparse_indices,
)

# Entry point (similar to ulysses_context_parallel and ring_context_parallel)
from .deepseek_v4_context_parallel import (
    DeepSeekV4CPContextParallel,
)

__all__ = [
    # Data types
    "DeepSeekV4CPAlignmentDescriptor",
    "DeepSeekV4CPCompressedKV",
    "DeepSeekV4CPCompressionCandidates",
    "DeepSeekV4CPCompressContext",
    "DeepSeekV4CPMetadata",
    "DeepSeekV4CPPendingCompressedKV",
    "DeepSeekV4CPPackedSeqMetadata",
    "DeepSeekV4CPRuntimeMetadata",
    "DeepSeekV4CPSMLAInputs",
    # Distributed communication
    "exchange_deepseek_v4_packed_previous_window",
    "exchange_deepseek_v4_previous_tails",
    "exchange_deepseek_v4_previous_window",
    # Compressed KV operations
    "compact_deepseek_v4_compressed_kv",
    "launch_deepseek_v4_allgather_compressed_kv",
    "prepare_deepseek_v4_compression_candidates_for_cp",
    "wait_deepseek_v4_compressed_kv",
    # SMLA input construction
    "align_deepseek_v4_cp_tensor",
    "build_deepseek_v4_causal_cmp_sparse_indices",
    "build_deepseek_v4_cmp_cu_seqlens",
    "build_deepseek_v4_cmp_residual_kv",
    "build_deepseek_v4_cp_packed_seq_metadata",
    "build_deepseek_v4_cp_smla_inputs",
    "build_deepseek_v4_owned_runtime_metadata",
    "flatten_deepseek_v4_cp_tensor_to_tnd",
    "remap_deepseek_v4_cp_sparse_indices",
    "validate_deepseek_v4_c4a_cmp_sparse_indices",
    # Entry point
    "DeepSeekV4CPContextParallel",
]
