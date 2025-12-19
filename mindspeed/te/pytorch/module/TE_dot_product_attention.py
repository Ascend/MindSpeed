# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from typing import Optional

import math
import torch
import torch_npu
from torch import Tensor
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_hierarchical_context_parallel_groups,
    get_tensor_model_parallel_group,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.packed_seq_params import PackedSeqParams
from mindspeed.core.context_parallel import AttnMaskType
from mindspeed.core.context_parallel import FusedScaleMaskSoftmax
from mindspeed.core.context_parallel import divide
from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
from mindspeed.core.transformer.flash_attention.alibi.alibi import AlibiForFusionAttnSingleton
from mindspeed.core.context_parallel import mpu as parallel_state
from mindspeed.model.transformer import get_attention_mask
from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
from mindspeed.core.context_parallel.ring_context_parallel.context_parallel_kv_cache import get_cache_policy
from mindspeed.core.context_parallel.ulysses_context_parallel.ulysses_context_parallel import ulyssesattn_context_parallel
from mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel import ringattn_context_parallel
from mindspeed.core.context_parallel.utils import get_scheduling_info
from mindspeed.core.context_parallel.adaptive_context_parallel.adaptive_context_parallel import adaptive_attn_context_parallel
from mindspeed.core.context_parallel.model_parallel_utils import (get_context_parallel_group_for_hybrid_ring,
                                           get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ring_rank,
                                           get_context_parallel_for_hybrid_ring_global_ranks,
                                           get_ring_ranks_for_intra_window,
                                           get_ring_ranks_for_inter_window_kv,
                                           get_ring_ranks_for_inter_window_dkv,
                                           get_ring_group_for_intra_window,
                                           get_ring_group_for_intra_window_send_recv_overlap)


try:
    from einops import rearrange
except ImportError:
    rearrange = None


class TEDotProductAttention(MegatronModule):
    """MindSpeed impl for the Transformer-Engine's `DotProductAttention` layer
    that also has "flash attention" enabled.

    Note that if Megatron's parallel_state has not been initialized yet, the
    tp_group and cp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group() and set_context_parallel_group().
    """

    cp_stream: torch.cuda.Stream = None

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None, #
        v_channels: Optional[int] = None, #
        cp_comm_type: str = "p2p",
        model_comm_pgs: ModelCommProcessGroups = None, #
    ):
        super().__init__(config=config)
        self.config = config
        self.cp_size = parallel_state.get_context_parallel_world_size()
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused for now

        kv_channels = k_channels * v_channels if (k_channels is not None and v_channels is not None) else self.config.kv_channels
        projection_size = kv_channels * self.config.num_attention_heads
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)

        if (attention_dropout or self.config.attention_dropout) is not None:
            self.attention_dropout = torch.nn.Dropout(
                self.config.attention_dropout if attention_dropout is None else attention_dropout
            )

        context_transdict = {
            'p2p': 'megatron_cp_algo',
            'a2a': 'ulysses_cp_algo',
            'a2a+p2p': 'hybrid_cp_algo'
        }
        if cp_comm_type not in context_transdict.keys():
            raise NotImplementedError(f'The cp type {cp_comm_type} is not support for MindSpeed TE!')

        if self.cp_size > 1:
            if context_transdict[cp_comm_type] != config.context_parallel_algo:
                print(f'MindSpeed CP {config.context_parallel_algo} not equal Megatron CP {cp_comm_type}! \
                    Please check CP type. Now runing CP type is {context_transdict[cp_comm_type]}')
        self.cp_algo = context_transdict[cp_comm_type]
        '''
        CP Comm group settings.
        In Mindspeed, p2p equals ring, a2a equals ulysses, and p2p+a2a equals hybird.
        '''
        if model_comm_pgs is None:
            model_comm_pgs = ModelCommProcessGroups(
                tp=get_tensor_model_parallel_group(check_initialized=False),
                cp=get_context_parallel_group(check_initialized=False),
                hcp=get_hierarchical_context_parallel_groups(check_initialized=False),
            )
        else:
            if not hasattr(model_comm_pgs, "tp"):
                raise AssertionError("TEDotProductAttention model_comm_pgs must have tp pg")
            if not hasattr(model_comm_pgs, "cp"):
                raise AssertionError("TEDotProductAttention model_comm_pgs must have cp pg")
            if cp_comm_type == "a2a+p2p":
                if not hasattr(model_comm_pgs, "hcp"):
                    raise AssertionError("TEDotProductAttention model_comm_pgs must have hierarchical cp pg")

        self.is_ulysses_algo = self.cp_algo == 'ulysses_cp_algo'

        # scale cal.
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)

        else:
            self.softmax_scale = softmax_scale

        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.softmax_scale /= coeff

        self.scale = self.softmax_scale

        self.cp_expanded_by_2d_tp = getattr(self.config, 'tp_2d', False) and getattr(self.config, 'tp_y', 1) > 1
        if self.cp_expanded_by_2d_tp:
            self.tp_y_cp_sz = TensorParallelYUnionCP().get_parallel_group_world_size()
        else:
            self.tp_y_cp_sz = self.config.context_parallel_size

        # add pse
        self.pse = None
        self.pse_type = self.config.alibi_fusion_attn_type

        if self.pse_type is None:
            self.pse_type = 1 # not use pse
        elif self.pse_type == 0:
            alibi = (
                AlibiForFusionAttnSingleton.get_alibi_tensor_for_fusion_attn(
                    self.config.seq_length,
                    self.config.num_attention_heads,
                    self.config.params_dtype,
                    self.config.alibi_diagonal_opposite,
                    1024
                )
            )
            self.pse = alibi
        elif self.pse_type == 2 or self.pse_type == 3:
            self.pse = (
                AlibiForFusionAttnSingleton.get_alibi_slopes_for_fusion_attn(
                    self.config.num_attention_heads
                )
            )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """Forward."""
        if attention_bias is not None:
            raise AssertionError("Attention bias is not supported for TEDotProductAttention.")

        if attention_mask is None and self.attn_mask_type == AttnMaskType.causal:
            if not getattr(self.config, 'is_llava', False):
                attention_mask = get_attention_mask()
                if self.config.attention_mask_type == 'causal':
                    self.config.sparse_mode = 2
                if getattr(self.config, 'reset_attention_mask', False):
                    if self.config.attention_mask_type == 'general':
                        self.config.sparse_mode = 2
                        if not (self.config.context_parallel_size == 1 or self.cp_algo == 'ulysses_cp_algo'):
                            self.config.sparse_mode = 1

        sparse_mode = self.config.sparse_mode

        if packed_seq_params is not None and not self.is_ulysses_algo:
            #TND
            _, n_head, _ = query.shape[0], query.shape[1], query.shape[2]
        else:
            _, bsz, n_head, head_dim = query.shape[0], query.shape[1], query.shape[2], query.shape[3]

        if packed_seq_params is not None and not self.is_ulysses_algo:
            # TND
            
            actual_seq_qlen = packed_seq_params.cu_seqlens_q.tolist()
            actual_seq_kvlen = packed_seq_params.cu_seqlens_kv.tolist()
            shape_order = 'TND'
        else:
            # SBH
            actual_seq_qlen = None if packed_seq_params is None else packed_seq_params.cu_seqlens_q.tolist()
            actual_seq_kvlen = None if packed_seq_params is None else packed_seq_params.cu_seqlens_kv.tolist()
            query, key, value = [rearrange(x, 's b h d -> s b (h d)') for x in [query, key, value]]
            shape_order = 'SBH'

        if attn_mask_type == AttnMaskType.no_mask:
            sparse_mode = 0  # default mask
    
        if (self.config.context_parallel_size > 1 and self.cp_algo == "ulysses_cp_algo"
                and self.config.context_parallel_kv_cache_policy):
            self.ulysses_comm_para['cache_policy'] = get_cache_policy(
                self.layer_number, self.config.context_parallel_kv_cache_policy, self.config.context_parallel_cache_interval
            )
            self.ulysses_comm_para['use_ulysses_allgather_kv'] = self.config.use_ulysses_allgather_kv

            attn_para = dict()
            attn_para['packed_seq_params'] = packed_seq_params
            attn_para['attention_mask'] = attention_mask
            attn_para['scale'] = self.scale
            attn_para['pre_tokens'] = self.config.pre_tockens
            attn_para['next_tokens'] = self.config.next_tockens
            attn_para['keep_prob'] = 1 - self.attention_dropout.p
            attn_para['sparse_mode'] = sparse_mode
            attn_para['n_head'] = n_head
            output = ulyssesattn_context_parallel(query, key, value, attn_para, self.ulysses_comm_para)

            return output

        if self.tp_y_cp_sz > 1 and self.cp_algo in ['megatron_cp_algo', 'hybrid_cp_algo',
                                                             'adaptive_cp_algo', 'hybrid_adaptive_cp_algo']:
            in_hybrid_mode = False
            if get_context_parallel_group_for_hybrid_ring(check_initialized=False) is not None:
                in_hybrid_mode = True

            if not in_hybrid_mode:
                if self.cp_expanded_by_2d_tp:
                    tp_y_cp = TensorParallelYUnionCP()
                    cp_group = tp_y_cp.group
                    cp_size = tp_y_cp.get_parallel_group_world_size()
                    rank = tp_y_cp.get_parallel_rank()
                    cp_global_ranks = tp_y_cp.global_ranks
                else:
                    cp_group = parallel_state.get_context_parallel_group()
                    cp_size = parallel_state.get_context_parallel_world_size()
                    rank = parallel_state.get_context_parallel_rank()
                    cp_global_ranks = parallel_state.get_context_parallel_global_ranks()
            else:
                cp_group = get_context_parallel_group_for_hybrid_ring()
                cp_size = get_context_parallel_for_hybrid_ring_world_size()
                rank = get_context_parallel_for_hybrid_ring_rank()
                cp_global_ranks = get_context_parallel_for_hybrid_ring_global_ranks()

            cp_para = dict()
            cp_para['megatron_cp_in_bnsd'] = self.config.megatron_cp_in_bnsd
            cp_para['causal'] = self.config.attention_mask_type == 'causal'
            cp_para['cp_group'] = cp_group
            cp_para['cp_size'] = cp_size
            cp_para['rank'] = rank

            if self.cp_algo in ['megatron_cp_algo', 'hybrid_cp_algo']:
                is_general_eod = ((getattr(self.config, 'attention_mask_type', None) == 'general') and (packed_seq_params is not None))
                if is_general_eod:
                    query, key, value = [rearrange(x, '(b s) n d -> s b (n d)', b=self.config.micro_batch_size) for x in [query, key, value]]
                cp_para['cp_global_ranks'] = cp_global_ranks
                if self.config.use_cp_send_recv_overlap:
                    if self.cp_expanded_by_2d_tp:
                        cp_para['cp_group_for_send_recv_overlap'] = tp_y_cp.overlap_group
                    else:
                        cp_para[
                            'cp_group_for_send_recv_overlap'] = parallel_state.get_context_parallel_group_for_send_recv_overlap()
                else:
                    cp_para['cp_group_for_send_recv_overlap'] = None
                cp_para['pse'] = self.pse
                cp_para['pse_type'] = self.pse_type

                if self.config.context_parallel_size > 1 and not getattr(self.config, 'tp_2d', False):
                    cp_para['cp_inner_ranks'] = get_ring_ranks_for_intra_window()
                    cp_para['cp_outer_ranks'] = get_ring_ranks_for_inter_window_kv()
                    cp_para['cp_dkv_outer_ranks'] = get_ring_ranks_for_inter_window_dkv()
                    cp_para['cp_group_for_intra_window'] = get_ring_group_for_intra_window()
                    cp_para[
                        'cp_group_for_intra_window_send_recv_overlap'] = get_ring_group_for_intra_window_send_recv_overlap()
                    cp_para['cache_policy'] = get_cache_policy(
                        self.layer_number, self.config.context_parallel_kv_cache_policy, self.config.context_parallel_cache_interval
                    )
                output = ringattn_context_parallel(query, key, value, n_head, cp_para, self.scale, attention_mask,
                                                   self.attention_dropout.p,
                                                   packed_seq_params)
                if is_general_eod:
                    output = rearrange(output, 's b (n d) -> (b s) n d', n=n_head)

            else:
                cp_para['scheduling_info'] = get_scheduling_info()
                output = adaptive_attn_context_parallel(query, key, value, n_head, cp_para, self.scale, attention_mask,
                                                        self.attention_dropout.p)

        else:
            # For EoD ulysses
            if packed_seq_params is not None:
                query, key, value = [rearrange(x, 's b (h d) -> (b s) h d', d=head_dim) for x in [query, key, value]]
                shape_order = 'TND'

            if self.config.use_fusion_attn_v2:
                output = npu_fusion_attention(
                    query, key, value, n_head, shape_order,
                    pse=self.pse,
                    padding_mask=None,
                    atten_mask=attention_mask,
                    scale=self.scale,
                    pse_type=self.pse_type,
                    pre_tokens=self.config.pre_tockens,
                    next_tokens=self.config.next_tockens,
                    keep_prob=1 - self.attention_dropout.p,
                    inner_precise=0,
                    sparse_mode=sparse_mode,
                    actual_seq_qlen=actual_seq_qlen,
                    actual_seq_kvlen=actual_seq_kvlen
                )[0]
            else:
                output = torch_npu.npu_fusion_attention(
                    query, key, value, n_head, shape_order,
                    pse=None,
                    padding_mask=None,
                    atten_mask=attention_mask,
                    scale=self.scale,
                    pre_tockens=self.config.pre_tockens,
                    next_tockens=self.config.next_tockens,
                    keep_prob=1 - self.attention_dropout.p,
                    inner_precise=0,
                    sparse_mode=sparse_mode,
                    actual_seq_qlen=actual_seq_qlen,
                    actual_seq_kvlen=actual_seq_kvlen
                )[0]

            if packed_seq_params is not None:
                output = rearrange(output, '(b s) h d -> s b (h d)', b=bsz)
                shape_order = 'TND'

        return output
