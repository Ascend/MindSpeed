# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import math
import torch
from torch import Tensor
import torch_npu
from mindspeed.core.tensor_parallel.tp_2d.utils import divide
from mindspeed.core.tensor_parallel.tp_2d.parallel_linear_2d import ParallelLinear2D
from mindspeed.core.tensor_parallel.tp_2d.parallel_state_2d import get_tensor_model_parallel_world_size_for_nd1_dim1
from mindspeed.core.tensor_parallel.tp_2d.group_api_2d import TPXCollectiveComm, TPXOverlapCollectiveComm, \
    TPYCollectiveComm, TPYOverlapCollectiveComm
from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
from mindspeed.core.context_parallel.ulysses_context_parallel import UlyssesContextAttention
from mindspeed.core.parallel_state import get_context_parallel_group_for_hybrid_ulysses
from mindspeed.core.context_parallel.context_parallel_kv_cache import get_cache_policy
from mindspeed.core.context_parallel.ulysses_context_parallel import ulyssesattn_context_parallel
from mindspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel
from mindspeed.core.context_parallel.utils import get_scheduling_info
from mindspeed.core.context_parallel.adaptive_context_parallel import adaptive_attn_context_parallel
from mindspeed.core.parallel_state import (get_context_parallel_group_for_hybrid_ring,
                                           get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ring_rank,
                                           get_context_parallel_for_hybrid_ring_global_ranks,
                                           get_ring_ranks_for_intra_window,
                                           get_ring_ranks_for_inter_window_kv,
                                           get_ring_ranks_for_inter_window_dkv,
                                           get_ring_group_for_intra_window,
                                           get_ring_group_for_intra_window_send_recv_overlap)
from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention

try:
    from einops import rearrange
except ImportError:
    rearrange = None
    
    
def self_attention_2d_init(
    self,
    config,
    submodules,
    layer_number,
    attn_mask_type,
    _initialize_affine_weight_gpu=None,
    **kwargs
    ):
    if config.tp_2d:
        attn_heads_split_num = get_tensor_model_parallel_world_size_for_nd1_dim1()
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, attn_heads_split_num)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, attn_heads_split_num)
        self.linear_qkv = ParallelLinear2D(
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            add_bias=self.config.add_bias_linear,
            skip_bias_add=True,
            ag_comm_intf=TPXCollectiveComm,
            ag_sd_rcv_overlap_comm_intf=TPXOverlapCollectiveComm,
            rs_comm_intf=TPYCollectiveComm,
            rs_sd_rcv_overlap_comm_intf=TPYOverlapCollectiveComm,
            enable_overlap_ag_with_matmul=False,
            enable_overlap_matmul_with_rs=False,
            partition_dim=0,
            enable_backward_overlap_ag_with_matmul=False,
            _initialize_affine_weight_gpu=_initialize_affine_weight_gpu
        )
        self.linear_proj = ParallelLinear2D(
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            add_bias=self.config.add_bias_linear,
            skip_bias_add=True,
            ag_comm_intf=TPYCollectiveComm,
            ag_sd_rcv_overlap_comm_intf=TPYOverlapCollectiveComm,
            rs_comm_intf=TPXCollectiveComm,
            rs_sd_rcv_overlap_comm_intf=TPXOverlapCollectiveComm,
            enable_overlap_ag_with_matmul=False,
            enable_overlap_matmul_with_rs=False,
            partition_dim=1,
            enable_backward_overlap_ag_with_matmul=config.enable_backward_overlap_ag_with_matmul,
            _initialize_affine_weight_gpu=_initialize_affine_weight_gpu
        )


def attention_init_impl(
    self,
    config,
    submodules,
    layer_number,
    attn_mask_type,
    attention_type,
    cp_comm_type,
    parallel_state=None,
    build_module_func=None
):
    self.config = config
    self.layer_number = layer_number
    self.attn_mask_type = attn_mask_type
    self.attention_type = attention_type

    # For normal attention without groups, num_query_groups == num_attention_heads,
    # so these two will be the same
    self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads
    self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

    # patch for tp-2d
    world_size = config.tp_x if config.tp_2d else parallel_state.get_tensor_model_parallel_world_size()
    # Per attention head and per partition values.
    self.hidden_size_per_attention_head = divide(
        self.query_projection_size, self.config.num_attention_heads
    )
    self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
    self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

    self.core_attention = build_module_func(
        submodules.core_attention,
        config=self.config,
        layer_number=self.layer_number,
        attn_mask_type=self.attn_mask_type,
        attention_type=self.attention_type,
        cp_comm_type=cp_comm_type,
    )

    self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'

    # Output.
    self.linear_proj = build_module_func(
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

    tp_y_cp_sz = config.tp_y * config.context_parallel_size
    if tp_y_cp_sz > 1 and getattr(config, 'context_parallel_algo', 'ulysses_cp_algo') == 'ulysses_cp_algo':
        if config.tp_2d:
            tp_y_cp = TensorParallelYUnionCP()
            ulysses_group = tp_y_cp.group
        else:
            ulysses_group = parallel_state.get_context_parallel_group()
        self.core_attention = UlyssesContextAttention(self.core_attention, ulysses_group)
