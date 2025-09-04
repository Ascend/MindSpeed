# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from functools import wraps
from typing import Optional

import torch
from torch import Tensor
from einops import rearrange

from megatron.training import get_args
from megatron.training.global_vars import get_args as get_global_args
from megatron.core import parallel_state
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
)
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.inference_params import InferenceParams
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import deprecate_inference_params

from megatron.core.transformer.multi_latent_attention import MLASelfAttention

from mindspeed.utils import get_position_ids, set_position_ids
from mindspeed.core.context_parallel.get_batch_utils import get_actual_seq_len, set_actual_seq_len
from mindspeed.core.context_parallel.rotary_pos_embedding_utils import get_pos_emb_on_this_cp_rank
from mindspeed.core.fusions.fused_rope import apply_rotary_pos_emb_bshd, apply_rotary_pos_emb


def _p2p_ops_eod(
        *,
        tensor_send_prev: Optional[torch.Tensor],
        tensor_recv_prev: Optional[torch.Tensor],
        tensor_send_next: Optional[torch.Tensor],
        tensor_recv_next: Optional[torch.Tensor],
        group: torch.distributed.ProcessGroup,
        prev_pipeline_rank: int,
        next_pipeline_rank: int,
):
    reqs = {}
    rank = get_pipeline_model_parallel_rank()
    even_send_odd_recv_group = group
    if get_pipeline_model_parallel_world_size() == 2:
        # Use the global process group for one of the two p2p communications
        # to allow the overlap of the independent communications.
        # Using the global process group is compatible because the pipeline-parallel
        # communications set the source and destination by global rank.
        even_recv_odd_send_group = torch.distributed.group.WORLD
    else:
        even_recv_odd_send_group = group

    prev_actual_seq_len = get_actual_seq_len()
    prev_position_ids = get_position_ids()

    tensor_length = None
    length_buffer = None

    args = get_args()
    bsz = args.micro_batch_size

    if tensor_send_next is not None:
        tensor_length = torch.tensor(prev_actual_seq_len.numel()).npu()

    if tensor_recv_prev is not None:
        length_buffer = torch.empty((), dtype=torch.int64, device=torch.cuda.current_device())

    if rank % 2 == 0:
        if tensor_length is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_length, dst=next_pipeline_rank, group=group,
            )
            reqs["send_next"] = send_next_req

        if length_buffer is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=length_buffer, src=prev_pipeline_rank, group=group,
            )
            reqs["recv_prev"] = recv_prev_req
    else:
        if length_buffer is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=length_buffer, src=prev_pipeline_rank, group=group,
            )
            reqs["recv_prev"] = recv_prev_req

        if tensor_length is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_length, dst=next_pipeline_rank, group=group,
            )
            reqs["send_next"] = send_next_req

    for req in reqs.values():
        req.wait()

    reqs = {}

    if get_pipeline_model_parallel_rank() % 2 == 0:
        if tensor_send_next is not None:
            req = torch.distributed.isend(
                tensor=prev_actual_seq_len, dst=next_pipeline_rank, group=even_send_odd_recv_group,
            )
            reqs["req"] = req

            req = torch.distributed.isend(
                tensor=prev_position_ids, dst=next_pipeline_rank, group=even_send_odd_recv_group,
            )
            reqs["req"] = req

            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=next_pipeline_rank, group=even_send_odd_recv_group,
            )
            reqs["send_next"] = send_next_req

        if tensor_recv_prev is not None:
            actual_seq_len_buffer = torch.empty([length_buffer.item()], dtype=torch.int64,
                                                device=torch.cuda.current_device())

            req = torch.distributed.irecv(
                tensor=actual_seq_len_buffer, src=prev_pipeline_rank, group=even_recv_odd_send_group,
            )
            reqs["req"] = req
            set_actual_seq_len(actual_seq_len_buffer)

            dynamic_seq_len = tensor_recv_prev.shape[0]
            # If SP on, sequence would be divided by tp_size
            if args.sequence_parallel:
                dynamic_seq_len *= args.tensor_model_parallel_size
            position_ids_buffer = torch.empty((dynamic_seq_len, bsz), dtype=torch.int64,
                                              device=torch.cuda.current_device())
            req = torch.distributed.irecv(
                tensor=position_ids_buffer, src=prev_pipeline_rank, group=even_recv_odd_send_group,
            )
            set_position_ids(position_ids_buffer)
            reqs["req"] = req

            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=prev_pipeline_rank, group=even_recv_odd_send_group,
            )
            reqs["recv_prev"] = recv_prev_req

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=prev_pipeline_rank, group=even_send_odd_recv_group,
            )
            reqs["send_prev"] = send_prev_req

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=next_pipeline_rank, group=even_recv_odd_send_group,
            )
            reqs["recv_next"] = recv_next_req

    else:
        if tensor_recv_prev is not None:
            actual_seq_len_buffer = torch.empty([length_buffer.item()], dtype=torch.int64,
                                                device=torch.cuda.current_device())

            req = torch.distributed.irecv(
                tensor=actual_seq_len_buffer, src=prev_pipeline_rank, group=even_send_odd_recv_group,
            )
            reqs["req"] = req
            set_actual_seq_len(actual_seq_len_buffer)

            dynamic_seq_len = tensor_recv_prev.shape[0]
            # If SP on, sequence would be divided by tp_size
            if args.sequence_parallel:
                dynamic_seq_len *= args.tensor_model_parallel_size
            position_ids_buffer = torch.empty((dynamic_seq_len, bsz), dtype=torch.int64,
                                              device=torch.cuda.current_device())
            req = torch.distributed.irecv(
                tensor=position_ids_buffer, src=prev_pipeline_rank, group=even_send_odd_recv_group,
            )
            set_position_ids(position_ids_buffer)
            reqs["req"] = req

            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=prev_pipeline_rank, group=even_send_odd_recv_group,
            )
            reqs["recv_prev"] = recv_prev_req

        if tensor_send_next is not None:
            req = torch.distributed.isend(
                tensor=prev_actual_seq_len, dst=next_pipeline_rank, group=even_recv_odd_send_group,
            )
            reqs["req"] = req

            req = torch.distributed.isend(
                tensor=prev_position_ids, dst=next_pipeline_rank, group=even_recv_odd_send_group,
            )
            reqs["req"] = req

            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=next_pipeline_rank, group=even_recv_odd_send_group,
            )
            reqs["send_next"] = send_next_req

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=next_pipeline_rank, group=even_send_odd_recv_group,
            )
            reqs["recv_next"] = recv_next_req

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=prev_pipeline_rank, group=even_recv_odd_send_group,
            )
            reqs["send_prev"] = send_prev_req
    return reqs


def attention_forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params=None,
):
    # For self attention we just duplicate the rotary_pos_emb if it isn't already
    if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
        rotary_pos_emb = (rotary_pos_emb,) * 2

    # =====================
    # Query, Key, and Value
    # =====================
    # Get the query, key and value tensors based on the type of attention -
    # self or cross attn.
    query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)
    bsz = query.shape[1]

    # ===================================================
    # Adjust key, value, and rotary_pos_emb for inference
    # ===================================================
    query, key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
        inference_context, query, key, value, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset
    )

    # ================================================
    # relative positional embedding (rotary embedding)
    # ================================================
    if rotary_pos_emb is not None:
        q_pos_emb, k_pos_emb = rotary_pos_emb

        if packed_seq_params is not None:
            cu_seqlens_q = packed_seq_params
            cu_seqlens_kv = packed_seq_params
        else:
            cu_seqlens_q = cu_seqlens_kv = None
        query = apply_rotary_pos_emb(
            query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q,
        )
        key = apply_rotary_pos_emb(
            key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv,
        )
    is_ulysses_algo = (getattr(self.config, 'context_parallel_algo', None) == 'ulysses_cp_algo')

    if packed_seq_params is not None and not is_ulysses_algo:
        query, key, value = [rearrange(x, 's b h d -> (b s) h d') for x in [query, key, value]]

    # ==================================
    # core attention computation
    # ==================================

    if self.checkpoint_core_attention and self.training:
        core_attn_out = self._checkpointed_attention_forward(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
    else:
        core_attn_out = self.core_attention(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )

    # =================
    # Output. [sq, b, h]
    # =================
    if packed_seq_params is not None and not is_ulysses_algo:
        core_attn_out = rearrange(core_attn_out, '(b s) h d -> s b (h d)', b=bsz)

    output, bias = self.linear_proj(core_attn_out)

    return output, bias


class MindSpeedMLASelfAttention(MLASelfAttention):
    """MLA Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def get_query_key_value_tensors(
            self,
            hidden_states,
            key_value_states=None,
            position_ids=None,
            packed_seq_params=None,
            inference_context=None,
            *,
            inference_params=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # s = sequence length, b = batch size, h = hidden size, n = num attention heads
        # Attention heads [s, b, n*h]
        assert (
                hidden_states.ndim == 3
        ), f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"

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
            packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len)

        if packed_seq_params is not None:
            cu_seqlens_q = packed_seq_params
            cu_seqlens_kv = packed_seq_params
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

            q_compressed = self.q_layernorm(q_compressed)
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
            if self.config.sequence_parallel:
                # kv_compressed:[s / TP, b, kv_lora_rank]
                kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        else:
            # kv_compressed:[s / TP, b, kv_lora_rank], k_pos_emb: [s / TP, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                # k_pos_emb: [s, b, qk_pos_emb_head_dim]
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb)

        kv_compressed = self.kv_layernorm(kv_compressed)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================
        def qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb):
            if self.config.q_lora_rank is not None:
                q, _ = self.linear_q_up_proj(q_compressed)
            else:
                # hidden_states:[s, b, 2048], q: [s, b, n * 192]
                q, _ = self.linear_q_proj(q_compressed)

            q_len, bsz, _ = q.size()

            # q: [s, b, n, 192]
            q = q.view(q_len, bsz, self.num_attention_heads_per_partition, self.q_head_dim)

            # kv: [s, b, 2048]
            kv, _ = self.linear_kv_up_proj(kv_compressed)

            # kv: [s, b, n, 256]
            kv = kv.view(
                q_len,
                bsz,
                self.num_attention_heads_per_partition,
                self.config.qk_head_dim + self.config.v_head_dim,
            )

            if inference_context is not None:
                # add offset to the sequence start for inference
                sequence_start = inference_context.sequence_len_offset
                sequence_end = sequence_start + q_len
                rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]
            else:
                # Shorten rotary_pos_emb to the sequence length when inference_params
                # is not provided. This makes sure we can run forward directly with
                # any sequence length. During training, the sequence length is always
                # the full rotary_pos_emb length.
                rotary_pos_emb = rotary_pos_emb[0:q_len]

            # [s, b, 64] -> [s, b, 1, 64]
            k_pos_emb = torch.unsqueeze(k_pos_emb, 2)

            # q: [s, b, n, 128], q_pos_emb: [s, b, n, 64]
            q_no_pe, q_pos_emb = torch.split(
                q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
            )

            # k_no_pe: [s, b, n, 128], value: [s, b, n, 128]
            k_no_pe, value = torch.split(
                kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1
            )

            # q_pos_emb: [s, b, n, 64], k_pos_emb:[s, b, 1, 64]
            q_pos_emb = apply_rotary_pos_emb(
                q_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
                mscale=mscale,
            )
            k_pos_emb = apply_rotary_pos_emb(
                k_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
                mscale=mscale,
            )

            # query: [s, b, n, 192]
            query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

            # key: [s, b, n, 192]
            k_pos_emb = k_pos_emb.expand(-1, -1, self.num_attention_heads_per_partition, -1)
            key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            return query, key, value

        if self.recompute_up_proj:
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            query, key, value = self.qkv_up_checkpoint.checkpoint(
                qkv_up_proj_and_rope_apply, q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )
        else:
            query, key, value = qkv_up_proj_and_rope_apply(
                q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )

        return query, key, value

    def forward(
            self,
            hidden_states,
            attention_mask,
            key_value_states=None,
            inference_context=None,
            rotary_pos_emb=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            attention_bias=None,
            packed_seq_params=None,
            position_ids=None,
            sequence_len_offset=None,
            *,
            inference_params=None,
    ):
        """Forward pass for multi-latent attention"""
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert (
                rotary_pos_cos is None and rotary_pos_sin is None
        ), "MLA does not support Flash Decoding"

        # hidden_states: [sq, b, h]

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]
        query, key, value = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
        )

        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        # rotary_pos_emb = None
        query, key, value, _, attn_mask_type = self._adjust_key_value_for_inference(
            inference_context, query, key, value, rotary_pos_emb=None
        )

        bsz = query.shape[1]
        is_ulysses_algo = (getattr(self.config, 'context_parallel_algo', None) == 'ulysses_cp_algo')
        if packed_seq_params is not None and not is_ulysses_algo:
            query, key, value = [rearrange(x, 's b h d -> (b s) h d') for x in [query, key, value]]

        # Currently, TE can only accept contiguous tensors for MLA
        query = query.contiguous()
        key = key.contiguous()
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
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                attn_mask_type=attn_mask_type,
            )

        if packed_seq_params is not None and not is_ulysses_algo:
            core_attn_out = rearrange(core_attn_out, '(b s) h d -> s b (h d)', b=bsz)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)

        return output, bias


def gpt_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        actual_seq_len = get_actual_seq_len()

        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=actual_seq_len,
            cu_seqlens_kv=actual_seq_len
        )

        actual_seq_len_list = actual_seq_len.tolist()
        max_actual_seq_len = actual_seq_len_list[0]
        for i in range(1, len(actual_seq_len_list)):
            max_actual_seq_len = max(max_actual_seq_len, actual_seq_len_list[i] - actual_seq_len_list[i - 1])
        packed_seq_params.max_seqlen_q = max_actual_seq_len
        packed_seq_params.max_seqlen_kv = max_actual_seq_len

        q_index, kv_index = compute_qkv_index(actual_seq_len_list)
        packed_seq_params.q_index = q_index
        packed_seq_params.kv_index = kv_index
        packed_seq_params.position_ids = get_position_ids()

        kwargs['packed_seq_params'] = packed_seq_params
        return fn(*args, **kwargs)

    return wrapper


def compute_qkv_index(seq_lens):
    args = get_global_args()
    if args.attention_mask_type == 'general' or get_ring_degree() == 1:
        return None, None

    full_indices = list(range(seq_lens[-1]))
    prev_eod_pos = 0
    kv_indices = []
    q_indices = []
    for eod_pos in seq_lens:
        mid = (eod_pos + prev_eod_pos) // 2
        kv_indices.extend(full_indices[prev_eod_pos:mid])
        q_indices.extend(full_indices[mid:eod_pos])
        prev_eod_pos = eod_pos

    kv_index = torch.tensor(kv_indices).cuda(non_blocking=True)
    q_index = torch.tensor(q_indices).cuda(non_blocking=True)

    return q_index, kv_index


def get_ring_degree():
    args = get_global_args()
    cp_size = args.context_parallel_size
    if cp_size == 1:
        return 1

    if args.context_parallel_algo == 'megatron_cp_algo':
        return cp_size
    elif args.context_parallel_algo == 'ulysses_cp_algo':
        return 1
    else:
        return args.ring_degree


def apply_rotary_pos_emb_thd(
        t: Tensor, cu_seqlens: Tensor, freqs: Tensor, rotary_interleaved: bool = False,
        multi_latent_attention: bool = False, mscale: float = 1.0
) -> Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """

    position_ids = cu_seqlens.position_ids
    block_size, bsz = position_ids.shape
    freqs = freqs[position_ids.view(-1)].reshape(block_size, bsz, 1, -1)

    return apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved, multi_latent_attention, mscale)


def Eod_get_rotary_seq_len(
        self,
        inference_context: BaseInferenceContext,
        transformer: TransformerBlock,
        transformer_input: Tensor,
        transformer_config: TransformerConfig,
        packed_seq_params: PackedSeqParams,
        inference_params: Optional[BaseInferenceContext] = None,
) -> float:
    """Function to get the rotary sequence length with Eod.

    Args:
        inference_params : Used during Inference time
        transformer (TransformerBlock): The transformer block (decoder/encoder) used
            by the model
        transformer_input (Tensor): Input tensor to the transformer
        transformer_config (TransformerConfig): Transformer config used by the model
        packed_seq_params (PackedSeqParams): Packed sequence params

    Returns:
        float: The rotary sequence length
    """

    if inference_params is not None:
        rotary_seq_len = inference_params.max_sequence_length
    else:
        if transformer is not None and transformer.input_tensor is not None:
            rotary_seq_len = transformer.input_tensor.size(0)
        else:
            rotary_seq_len = transformer_input.size(0)

        if transformer_config.sequence_parallel:
            rotary_seq_len *= transformer_config.tensor_model_parallel_size

    rotary_seq_len *= transformer_config.context_parallel_size

    return rotary_seq_len


def rotary_forward(self, max_seq_len: int, offset: int = 0, packed_seq: bool = False) -> Tensor:
    """Forward pass of RoPE embedding.

    Args:
        max_seq_len (int): Maximum size of sequence
        offset (int, optional): _description_. Defaults to 0.
        packed_seq (bool, optional): Whether to use packed sequence. Defaults to False.

    Returns:
        Tensor: Embeddings after applying RoPE.
    """
    if self.inv_freq.device.type == 'cpu':
        # move `inv_freq` to GPU once at the first micro-batch forward pass
        self.inv_freq = self.inv_freq.to(device=torch.cuda.current_device())
    seq = (
            torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            + offset
    )

    if self.seq_len_interpolation_factor is not None:
        seq *= 1 / self.seq_len_interpolation_factor

    freqs = torch.outer(seq, self.inv_freq)
    # first part even vector components, second part odd vector components,
    #  2 * dim in dimension size
    if not self.rotary_interleaved:
        emb = torch.cat((freqs, freqs), dim=-1)
    else:
        emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
            freqs.shape[0], -1
        )
    # emb [seq_length, .., dim]
    emb = emb[:, None, None, :]

    return emb
