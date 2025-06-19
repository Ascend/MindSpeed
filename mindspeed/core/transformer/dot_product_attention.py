# coding=utf-8
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import math
from functools import wraps
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_npu
from torch import Tensor
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.utils import divide
from megatron.training import get_args
from megatron.core import mpu, parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from mindspeed.core.models.common.embeddings.rotary_pos_embedding import (
    yarn_get_mscale,
)  # noqa
from mindspeed.core.context_parallel.ring_context_parallel import (
    ringattn_context_parallel,
)
from mindspeed.core.context_parallel.ulysses_context_parallel import (
    ulyssesattn_context_parallel,
)
from mindspeed.core.context_parallel.context_parallel_kv_cache import (
    get_cache_policy,
)  # noqa
from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
from mindspeed.model.alibi_mask import AlibiForFusionAttnSingleton
from mindspeed.core.parallel_state import (
    get_context_parallel_group_for_hybrid_ring,
    get_context_parallel_for_hybrid_ring_world_size,
    get_context_parallel_for_hybrid_ring_rank,
    get_context_parallel_for_hybrid_ring_global_ranks,
    get_ring_ranks_for_intra_window,
    get_ring_ranks_for_inter_window_kv,
    get_ring_ranks_for_inter_window_dkv,
    get_ring_group_for_intra_window,
    get_ring_group_for_intra_window_send_recv_overlap,
)
from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
from mindspeed.model.transformer import get_attention_mask
from mindspeed.utils import get_actual_seq_len
from mindspeed.core.context_parallel.adaptive_context_parallel import (
    adaptive_attn_context_parallel,
)
from mindspeed.core.context_parallel.utils import get_scheduling_info

try:
    from einops import rearrange, repeat
except ImportError:
    rearrange = None


def dot_product_attention_init(
    self,
    config: TransformerConfig,
    layer_number: int,
    attn_mask_type: AttnMaskType,
    attention_type: str,
    attention_dropout: float = None,
):
    cp_size = config.context_parallel_size
    config.context_parallel_size = 1

    super(DotProductAttention, self).__init__(config=config)
    assert (
        self.config.context_parallel_size == 1
    ), "Context parallelism is only supported by TEDotProductAttention!"

    assert (
        self.config.window_size is None
    ), "Sliding Window Attention is only supported by TEDotProductAttention!"

    self.layer_number = max(1, layer_number)
    self.attn_mask_type = attn_mask_type
    self.attention_type = attention_type  # unused for now

    projection_size = self.config.kv_channels * self.config.num_attention_heads
    args = get_args()
    # Per attention head and per partition values.
    world_size = (
        args.tp_x
        if args.tp_2d
        else parallel_state.get_tensor_model_parallel_world_size()
    )
    self.hidden_size_per_partition = divide(projection_size, world_size)
    self.hidden_size_per_attention_head = divide(
        projection_size, config.num_attention_heads
    )
    self.num_attention_heads_per_partition = divide(
        self.config.num_attention_heads, world_size
    )
    self.num_query_groups_per_partition = divide(
        self.config.num_query_groups, world_size
    )

    coeff = None
    self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
    if self.config.apply_query_key_layer_scaling:
        coeff = self.layer_number
        self.norm_factor *= coeff

    self.scale_mask_softmax = FusedScaleMaskSoftmax(
        input_in_fp16=self.config.fp16,
        input_in_bf16=self.config.bf16,
        attn_mask_type=self.attn_mask_type,
        scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
        mask_func=attention_mask_func,
        softmax_in_fp32=self.config.attention_softmax_in_fp32,
        scale=coeff,
    )

    # Dropout. Note that for a single iteration, this layer will generate
    # different outputs on different number of parallel partitions but
    # on average it should not be partition dependent.
    self.attention_dropout = torch.nn.Dropout(
        self.config.attention_dropout
        if attention_dropout is None
        else attention_dropout
    )

    config.context_parallel_size = cp_size

    # add pse
    self.pse = None
    self.pse_type = args.alibi_fusion_attn_type

    if args.multi_head_latent_attention:
        self.scale_mask_softmax.scale = True
        self.hidden_size_per_partition = (
            config.num_attention_heads * args.v_head_dim
        )  # noqa
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.softmax_scale = self.q_head_dim ** (-0.5)

        if args.rope_scaling_type is not None:
            mscale_all_dim = (
                args.rope_scaling_mscale_all_dim
                if args.rope_scaling_mscale_all_dim
                else 0
            )
            scaling_factor = args.rope_scaling_factor

            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.norm_factor = 1.0 / self.softmax_scale

    if self.pse_type is None:
        self.pse_type = 1  # not use pse
    elif self.pse_type == 0:
        alibi = AlibiForFusionAttnSingleton.get_alibi_tensor_for_fusion_attn(
            args.seq_length,
            config.num_attention_heads,
            config.params_dtype,
            args.alibi_diagonal_opposite,
            1024,
        )
        self.pse = alibi
    elif self.pse_type == 2 or self.pse_type == 3:
        self.pse = AlibiForFusionAttnSingleton.get_alibi_slopes_for_fusion_attn(  # noqa
            config.num_attention_heads
        )

    self.scale = (
        1.0 / math.sqrt(self.hidden_size_per_attention_head)
        if self.scale_mask_softmax.scale is None
        else self.softmax_scale
    )
    self.attention_starategy = choose_attention_strategy(self)


def choose_attention_strategy(self):
    """Choose the attention strategy based on the configuration."""
    args = get_args()
    cp_expanded_by_2d_tp = args.tp_2d and args.tp_y > 1

    if cp_expanded_by_2d_tp:
        tp_y_cp_sz = TensorParallelYUnionCP().get_parallel_group_world_size()
    else:
        tp_y_cp_sz = self.config.context_parallel_size

    # enable context parallel
    if tp_y_cp_sz > 1:
        if (
            args.context_parallel_algo == "ulysses_cp_algo"
            and args.context_parallel_kv_cache_policy
        ):
            return UlysessAttentionStrategy(self)
        if args.context_parallel_algo in (
            "megatron_cp_algo",
            "hybrid_cp_algo",
        ):
            return RingAttentionStrategy(self)
        elif args.context_parallel_algo in (
            "adaptive_cp_algo",
            "hybrid_adaptive_cp_algo",
        ):
            return AdaptiveAttentionStrategy(self)

    # disable context parallel
    if getattr(args, "use_fusion_attn_v2", False):
        return FlashAttentionV2Strategy(self)
    if getattr(self.config, "inference_attention", False):
        return InferenceAttentionStrategy(self)
    if getattr(self.config, "use_unpad_input_attention", False):
        return UnPadInputAttentionStrategy(self)
    if getattr(self.config, "use_general_mask_attention", False):
        return FlashAttentionStrategy(self, attention_mask_type="general")
    return FlashAttentionStrategy(self)


def dot_product_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        if self.config.num_query_groups is None:
            self.config.num_query_groups = self.config.num_attention_heads
        self.num_attention_heads_per_partition = (
            self.config.num_attention_heads
            * self.num_query_groups_per_partition
            // self.config.num_query_groups
        )

    return wrapper


def dot_product_attention_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type,
        packed_seq_params,
    ):
        if (
            attention_mask is None
            and self.attn_mask_type == AttnMaskType.causal  # type: ignore
        ):
            attention_mask = get_attention_mask()
        if get_args().use_flash_attn:
            return dot_product_attention_forward(
                self,
                query,
                key,
                value,
                attention_mask,
                attn_mask_type,
                packed_seq_params,
            )
        return fn(
            self,
            query,
            key,
            value,
            attention_mask,
            attn_mask_type,
            packed_seq_params,
        )

    return wrapper


def dot_product_attention_forward(
    self: DotProductAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor,
    attn_mask_type: AttnMaskType,
    packed_seq_params: Optional[PackedSeqParams] = None,
):
    """forward function for dot product attention."""

    return self.attention_starategy.forward(
        query,
        key,
        value,
        attention_mask,
        attn_mask_type,
        packed_seq_params,
    )


@dataclass
class AttentionArgs:
    """Key arguments for attention strategy."""

    seq_len: int
    batch_size: int
    n_head: int
    head_dim: int
    packed_seq_params: Optional[PackedSeqParams] = None
    new_shape_qkv: Optional[dict] = None
    actual_seq_qlen: Optional[list] = None
    actual_seq_kvlen: Optional[list] = None
    sparse_mode: int = 0
    attention_mask: Optional[Tensor] = None


class AttentionStrategy:
    """Base class for attention strategy."""

    def __init__(
        self,
        attention: DotProductAttention,
        attention_mask_type: str = "causal",
    ):
        self._attention = attention
        self._args = get_args()
        self._cp_expanded_by_2d_tp = self._args.tp_2d and self._args.tp_y > 1
        self._attn_mask_type = attention_mask_type

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        """forward method to be implemented by subclasses.
        Args:
            query (Tensor): Tensor of query.
            key (Tensor): Tensor of key.
            value (Tensor): Tensor of value.
            attention_mask (Tensor): Tensor of attention mask.
            attn_mask_type (AttnMaskType): attention mask type.
            packed_seq_params (Optional[PackedSeqParams], optional):
                parameters to TEDotProductAttention and fused rope kernels
                for the `thd` (packed) sequence format. Defaults to None.

        Returns:
            Tensor: Output tensor after applying attention.
        """
        raise NotImplementedError("forward method not implemented.")

    def calc_cp_params(self):
        """Calculate context parallel parameters."""
        in_hybrid_mode = False

        if (
            get_context_parallel_group_for_hybrid_ring(check_initialized=False)
            is not None
        ):
            in_hybrid_mode = True
        cp_group_for_send_recv_overlap = None
        if not in_hybrid_mode:
            if self._cp_expanded_by_2d_tp:
                tp_y_cp = TensorParallelYUnionCP()
                cp_group = tp_y_cp.group
                cp_size = tp_y_cp.get_parallel_group_world_size()
                rank = tp_y_cp.get_parallel_rank()
                cp_global_ranks = tp_y_cp.global_ranks
                if self._args.use_cp_send_recv_overlap:
                    cp_group_for_send_recv_overlap = tp_y_cp.overlap_group
            else:
                cp_group = mpu.get_context_parallel_group()
                cp_size = mpu.get_context_parallel_world_size()
                rank = mpu.get_context_parallel_rank()
                cp_global_ranks = mpu.get_context_parallel_global_ranks()
                if self._args.use_cp_send_recv_overlap:
                    cp_group_for_send_recv_overlap = (
                        mpu.get_context_parallel_group_for_send_recv_overlap()
                    )

        else:
            cp_group = get_context_parallel_group_for_hybrid_ring()
            cp_size = get_context_parallel_for_hybrid_ring_world_size()
            rank = get_context_parallel_for_hybrid_ring_rank()
            cp_global_ranks = (
                get_context_parallel_for_hybrid_ring_global_ranks()
            )  # noqa
        cp_para = dict()
        cp_para["megatron_cp_in_bnsd"] = (
            self._attention.config.megatron_cp_in_bnsd
        )  # noqa
        cp_para["causal"] = self._args.attention_mask_type == "causal"
        cp_para["cp_group"] = cp_group
        cp_para["cp_size"] = cp_size
        cp_para["rank"] = rank
        cp_para["cp_global_ranks"] = cp_global_ranks
        cp_para["cp_group_for_send_recv_overlap"] = (
            cp_group_for_send_recv_overlap  # noqa
        )
        cp_para["pse"] = self._attention.pse
        cp_para["pse_type"] = self._attention.pse_type
        return cp_para

    def update_packed_seq_params(self, packed_seq_params: PackedSeqParams):
        """Update packed sequence parameters."""
        actual_seq_len = get_actual_seq_len()
        if actual_seq_len is not None:
            if packed_seq_params is None:
                # Create new PackedSeqParams if not provided
                packed_seq_params = PackedSeqParams(
                    cu_seqlens_kv=actual_seq_len, cu_seqlens_q=actual_seq_len
                )
            else:
                # Update packed_seq_params with actual sequence lengths
                packed_seq_params.cu_seqlens_kv = actual_seq_len
                packed_seq_params.cu_seqlens_q = actual_seq_len
        return packed_seq_params

    def get_actual_seq_len(self, packed_seq_params: PackedSeqParams) -> tuple:
        """Get actual sequence lengths from packed sequence parameters."""
        if packed_seq_params is not None:
            actual_seq_qlen = (
                packed_seq_params.cu_seqlens_q.tolist()
                if isinstance(packed_seq_params.cu_seqlens_q, Tensor)
                else packed_seq_params.cu_seqlens_q
            )
            actual_seq_kvlen = (
                packed_seq_params.cu_seqlens_kv.tolist()
                if isinstance(packed_seq_params.cu_seqlens_q, Tensor)
                else packed_seq_params.cu_seqlens_kv
            )
        else:
            actual_seq_qlen = None
            actual_seq_kvlen = None
        return actual_seq_qlen, actual_seq_kvlen

    def get_reshape_qkv(
        self,
        packed_seq_params: PackedSeqParams,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> dict:
        """Reshape query, key, and value tensors
        based on packed sequence parameters.
        """
        if (
            getattr(self._attention.config, "use_repeat_kv", False)
            and self._attention.num_attention_heads_per_partition
            // self._attention.num_query_groups_per_partition
            > 1
        ):
            key = key.repeat_interleave(
                self._attention.num_attention_heads_per_partition
                // self._attention.num_query_groups_per_partition,
                dim=2,
            )
            value = value.repeat_interleave(
                self._attention.num_attention_heads_per_partition
                // self._attention.num_query_groups_per_partition,
                dim=2,
            )

        if packed_seq_params is not None:  # TND
            query, key, value = [
                rearrange(x, "s b h d -> (b s) h d")
                for x in [
                    query,
                    key,
                    value,
                ]
            ]
            shape_order = "TND"
        else:  # SBH
            query, key, value = [
                rearrange(x, "s b h d -> s b (h d)")
                for x in [
                    query,
                    key,
                    value,
                ]
            ]
            shape_order = "SBH"
        return {
            "query": query,
            "key": key,
            "value": value,
            "shape_order": shape_order,
        }

    def calc_sparse_mode(self, attn_mask_type: AttnMaskType) -> int:
        """Calculate sparse mode based on attention mask type."""
        sparse_mode = self._args.sparse_mode
        if attn_mask_type == AttnMaskType.no_mask:
            sparse_mode = 0  # default mask
        return sparse_mode

    def calc_attention_mask(
        self,
        attention_mask: Tensor,
        actual_seq_qlen: Optional[list],
        seq_len: int,
    ) -> Tensor:
        """Calculate attention mask based on the attention mask type."""
        if self._attn_mask_type == "general" or attention_mask is None:
            return None

        # when attention_mask is invalid, we create a causal mask
        if (
            attention_mask.dtype not in (torch.bool, torch.uint8)
            or attention_mask.dim() < 2
            or attention_mask.shape[-1] != attention_mask.shape[-2]
        ):
            if actual_seq_qlen:
                seq_len = max(actual_seq_qlen)
            return torch.triu(
                torch.ones(
                    [seq_len, seq_len],
                    dtype=torch.bool,
                    device=torch.cuda.current_device(),
                ),
                diagonal=1,
            )
        return attention_mask

    def calc_key_attention_args(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        packed_seq_params: Optional[PackedSeqParams] = None,
        reshape: bool = False,
    ) -> AttentionArgs:
        """Calculate key arguments for attention."""
        seq_len, batch_size, n_head, head_dim = (
            query.shape[0],
            query.shape[1],
            query.shape[2],
            query.shape[3],
        )
        packed_seq_params = self.update_packed_seq_params(packed_seq_params)
        new_shape_qkv = None
        if reshape:
            new_shape_qkv = self.get_reshape_qkv(
                packed_seq_params,
                query,
                key,
                value,
            )
        actual_seq_qlen, actual_seq_kvlen = self.get_actual_seq_len(
            packed_seq_params,
        )
        attention_mask = self.calc_attention_mask(
            attention_mask,
            actual_seq_qlen,
            seq_len,
        )
        sparse_mode = self.calc_sparse_mode(attn_mask_type)
        return AttentionArgs(
            seq_len=seq_len,
            batch_size=batch_size,
            n_head=n_head,
            head_dim=head_dim,
            packed_seq_params=packed_seq_params,
            new_shape_qkv=new_shape_qkv,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            sparse_mode=sparse_mode,
            attention_mask=attention_mask,
        )


class UlysessAttentionStrategy(AttentionStrategy):
    """Implementation of Ulysses attention strategy."""

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = "causal",
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """forward function for Ulysses attention strategy."""
        self._attention.ulysses_comm_para["cache_policy"] = get_cache_policy(
            self._attention.layer_number,
            self._args.context_parallel_kv_cache_policy,
            self._args.context_parallel_cache_interval,
        )
        self._attention.ulysses_comm_para["use_ulysses_allgather_kv"] = (
            self._args.use_ulysses_allgather_kv
        )

        attn_para = dict()
        attn_para["packed_seq_params"] = packed_seq_params
        attn_para["attention_mask"] = attention_mask
        attn_para["scale"] = self._attention.scale
        attn_para["pre_tokens"] = self._args.pre_tokens
        attn_para["next_tokens"] = self._args.next_tokens
        attn_para["keep_prob"] = 1 - self._attention.attention_dropout.p
        attn_para["sparse_mode"] = self.calc_sparse_mode(attn_mask_type)
        output = ulyssesattn_context_parallel(
            query, key, value, attn_para, self._attention.ulysses_comm_para
        )

        return output


class RingAttentionStrategy(AttentionStrategy):
    """Implementation of ring attention strategy."""

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """forward function for ring attention strategy."""
        attention_args = self.calc_key_attention_args(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type,
            packed_seq_params,
        )
        query, key, value = [
            rearrange(x, "s b h d -> s b (h d)") for x in [query, key, value]
        ]
        cp_para = self.calc_cp_params()
        packed_seq_params = self.update_packed_seq_params(packed_seq_params)
        if (
            self._attention.config.context_parallel_size > 1
            and not self._args.tp_2d  # type: ignore
        ):
            cp_para["cp_inner_ranks"] = get_ring_ranks_for_intra_window()
            cp_para["cp_outer_ranks"] = get_ring_ranks_for_inter_window_kv()
            cp_para["cp_dkv_outer_ranks"] = (
                get_ring_ranks_for_inter_window_dkv()
            )  # noqa
            cp_para["cp_group_for_intra_window"] = (
                get_ring_group_for_intra_window()
            )  # noqa
            cp_para["cp_group_for_intra_window_send_recv_overlap"] = (
                get_ring_group_for_intra_window_send_recv_overlap()
            )

        cp_para["cache_policy"] = get_cache_policy(
            self._attention.layer_number,
            self._args.context_parallel_kv_cache_policy,
            self._args.context_parallel_cache_interval,
        )

        output = ringattn_context_parallel(
            query,
            key,
            value,
            attention_args.n_head,
            cp_para,
            self._attention.scale,
            attention_mask,
            self._attention.attention_dropout.p,
            packed_seq_params,
        )

        return output


class AdaptiveAttentionStrategy(AttentionStrategy):
    """Implementation of adaptive attention strategy."""

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """forward function for adaptive attention strategy."""
        attention_args = self.calc_key_attention_args(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type,
            packed_seq_params,
        )
        query, key, value = [
            rearrange(x, "s b h d -> s b (h d)") for x in [query, key, value]
        ]
        cp_para = self.calc_cp_params()
        cp_para["scheduling_info"] = get_scheduling_info()

        output = adaptive_attn_context_parallel(
            query,
            key,
            value,
            attention_args.n_head,
            cp_para,
            self._attention.scale,
            attention_mask,
            self._attention.attention_dropout.p,
        )

        return output


class FlashAttentionV2Strategy(AttentionStrategy):
    """Implementation of flash attention v2 strategy."""

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """forward function for flash attention v2 strategy."""
        attention_args = self.calc_key_attention_args(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type,
            packed_seq_params,
            reshape=True,
        )
        output = npu_fusion_attention(
            attention_args.new_shape_qkv["query"],
            attention_args.new_shape_qkv["key"],
            attention_args.new_shape_qkv["value"],
            attention_args.n_head,
            attention_args.new_shape_qkv["shape_order"],
            pse=None,
            padding_mask=None,
            atten_mask=attention_args.attention_mask,
            scale=self._attention.scale,
            pre_tokens=self._args.pre_tockens,
            next_tokens=self._args.next_tockens,
            keep_prob=1 - self._attention.attention_dropout.p,
            inner_precise=0,
            sparse_mode=attention_args.sparse_mode,
            actual_seq_qlen=attention_args.actual_seq_qlen,
            actual_seq_kvlen=attention_args.actual_seq_kvlen,
        )[0]

        return output


class FlashAttentionStrategy(AttentionStrategy):
    """Implementation of flash attention strategy."""

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """forward function for flash attention strategy."""
        attention_args = self.calc_key_attention_args(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type,
            packed_seq_params,
            reshape=True,
        )
        output = torch_npu.npu_fusion_attention(
            attention_args.new_shape_qkv["query"],
            attention_args.new_shape_qkv["key"],
            attention_args.new_shape_qkv["value"],
            attention_args.n_head,
            attention_args.new_shape_qkv["shape_order"],
            pse=None,
            padding_mask=None,
            atten_mask=attention_args.attention_mask,
            scale=self._attention.scale,
            pre_tockens=self._args.pre_tockens,
            next_tockens=self._args.next_tockens,
            keep_prob=1 - self._attention.attention_dropout.p,
            inner_precise=0,
            sparse_mode=attention_args.sparse_mode,
            actual_seq_qlen=attention_args.actual_seq_qlen,
            actual_seq_kvlen=attention_args.actual_seq_kvlen,
        )[0]
        if attention_args.packed_seq_params is not None:
            output = rearrange(
                output,
                "(b s) h d -> s b (h d)",
                s=attention_args.seq_len,
                b=attention_args.batch_size,
            )
        return output


class InferenceAttentionStrategy(AttentionStrategy):
    """Implementation of inference attention strategy."""

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """forward function for inference attention strategy."""
        _seq_len, bsz, _n_head, _head_dim = (  # noqa
            query.shape[0],
            query.shape[1],
            query.shape[2],
            query.shape[3],
        )
        query = query.transpose(0, 1).contiguous()  # [b s h d]
        key = key.transpose(0, 1).contiguous()
        value = value.transpose(0, 1).contiguous()
        if query.shape[1] == 1:
            attention_mask_npu = None
        else:
            attention_mask_npu = torch.triu(
                torch.ones(
                    [query.shape[1], key.shape[1]],
                    dtype=torch.bool,
                    device=query.device,
                ),
                diagonal=1,
            )

        attn_output = torch_npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            pse_shift=None,
            atten_mask=attention_mask_npu,
            actual_seq_lengths=[query.shape[1]],
            actual_seq_lengths_kv=[key.shape[1]],
            num_heads=query.shape[2],
            num_key_value_heads=key.shape[2],
            scale=1.0 / math.sqrt(query.shape[-1]),
            input_layout="BSND",
        )[0]
        attn_output = rearrange(
            attn_output, "b s h d -> s b (h d)", s=query.shape[1], b=bsz
        )
        return attn_output


class IndexFirstAxis(torch.autograd.Function):
    """Implementation of indexing the first axis of a tensor."""

    @staticmethod
    def forward(ctx, input_tensor, indices):
        """Forward method to index the first axis of a tensor."""
        ctx.save_for_backward(indices)
        ctx.first_axis_dim, other_shape = input_tensor.shape[0], input_tensor.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(input_tensor, "b ... -> b (...)"),
            0,
            repeat(indices, "z -> z d", d=second_dim),
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward method to compute gradients for indexed tensor."""
        (indices,) = ctx.saved_tensors
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(
            0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output
        )
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    """Implementation of indexing and putting values
    in the first axis of a tensor.
    """

    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        """Forward method to index and put values in the first axis."""
        ctx.save_for_backward(indices)
        output = torch.zeros(
            first_axis_dim,
            *values.shape[1:],
            device=values.device,
            dtype=values.dtype,
        )
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward method to compute gradients for indexed tensor."""
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


class UnPadInputAttentionStrategy(AttentionStrategy):
    """Implementation of unpad input attention strategy."""

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """forward function for unpad input attention strategy."""

        seq_length, bsz, n_head, _head_dim = (  # noqa
            query.shape[0],
            query.shape[1],
            query.shape[2],
            query.shape[3],
        )
        query = query.transpose(0, 1).contiguous()
        key = key.transpose(0, 1).contiguous()
        value = value.transpose(0, 1).contiguous()

        # this is for mm qwenvl special inference handling logic.
        if attention_mask is not None and 0 not in attention_mask:
            attention_mask = None
        if attention_mask is None:
            attention_mask_npu = torch.triu(
                torch.ones(
                    [query.shape[1], key.shape[1]],
                    dtype=torch.bool,
                    device=query.device,
                ),
                diagonal=1,
            )
            attn_output = torch_npu.npu_fusion_attention(
                query,
                key,
                value,
                n_head,
                "BSND",
                keep_prob=1.0,
                scale=1.0 / math.sqrt(query.shape[-1]),
                atten_mask=attention_mask_npu,
            )[0]
            attn_output = rearrange(
                attn_output,
                "b s h d -> s b (h d)",
                s=seq_length,
                b=bsz,
            )
            return attn_output

        # unpad input handling logic for mm qwenvl
        query, key, value, indices_q, cu_seq_lens, max_seq_lens = self._unpad_input(
            query, key, value, attention_mask, seq_length
        )
        attention_mask_npu = torch.triu(
            torch.ones(
                [max_seq_lens, max_seq_lens],
                dtype=torch.bool,
                device=query.device,
            ),
            diagonal=1,
        )
        attn_output_unpad = torch_npu.npu_fusion_attention(
            query,
            key,
            value,
            n_head,
            pse=None,
            padding_mask=None,
            atten_mask=attention_mask_npu,
            scale=1.0 / math.sqrt(query.shape[-1]),
            keep_prob=1,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_seq_lens[1:].cpu().numpy().tolist()),
            actual_seq_kvlen=tuple(cu_seq_lens[1:].cpu().numpy().tolist()),
            pre_tockens=self._args.pre_tockens,
            next_tockens=self._args.next_tockens,
            sparse_mode=self.calc_sparse_mode(attn_mask_type),
        )[0]
        attn_output = self._pad_input(
            attn_output_unpad,
            indices_q,
            bsz,
            seq_length,
        )
        attn_output = rearrange(
            attn_output, "b s h d -> s b (h d)", s=seq_length, b=bsz
        )
        return attn_output

    def _unpad_input(
        self,
        query_state: torch.Tensor,
        key_state: torch.Tensor,
        value_state: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
    ):
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(
            attention_mask.flatten(),
            as_tuple=False,
        ).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
        )
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_state.shape
        key_state = index_first_axis(
            key_state.reshape(
                batch_size * kv_seq_len,
                num_key_value_heads,
                head_dim,
            ),
            indices,
        )
        value_state = index_first_axis(
            value_state.reshape(
                batch_size * kv_seq_len,
                num_key_value_heads,
                head_dim,
            ),
            indices,
        )
        if query_length != kv_seq_len:
            raise AssertionError(
                "Only support query_length that equals to kv_seq_len!",
            )
        query_state = index_first_axis(
            query_state.reshape(batch_size * kv_seq_len, -1, head_dim),
            indices,
        )

        return (
            query_state,
            key_state,
            value_state,
            indices,
            cu_seqlens,
            max_seqlen_in_batch,
        )

    @staticmethod
    def _pad_input(hidden_states, indices, batch, seqlen):
        output = index_put_first_axis(hidden_states, indices, batch * seqlen)
        return rearrange(output, "(b s) ... -> b s ...", b=batch)
