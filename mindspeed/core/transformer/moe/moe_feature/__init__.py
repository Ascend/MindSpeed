# Copyright (c) 2025; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler, TopKRouter
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.mlp import MLPSubmodules, MLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe import grouped_gemm_util
from megatron.core.transformer.spec_utils import build_module, ModuleSpec

from megatron.core.transformer.moe.moe_utils import (
    permute,
    unpermute,
    save_to_aux_losses_tracker,
    sort_chunks_by_idxs,
    get_capacity,
    topk_softmax_with_capacity
)

from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    linear_with_grad_accumulation_and_async_allreduce,
    linear_with_frozen_weight
)

from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
    _reduce_scatter_along_first_dim,
    _gather_along_first_dim,
    _split_along_first_dim
)

from megatron.core.tensor_parallel.utils import VocabUtility, divide, split_tensor_along_last_dim
from megatron.core.utils import (
    make_tp_sharded_tensor_for_checkpoint,
    prepare_input_tensors_for_wgrad_compute
)
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.moe.moe_layer import MoELayer, BaseMoELayer, BaseMoELayer as MegatronBaseMoeLayer
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP, set_tensor_grad_fn_sequence_sr
from megatron.core.transformer.moe.experts import GroupedMLP
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.moe.token_dispatcher import MoEAllGatherTokenDispatcher
from megatron.training import get_args
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import make_viewless_tensor
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl