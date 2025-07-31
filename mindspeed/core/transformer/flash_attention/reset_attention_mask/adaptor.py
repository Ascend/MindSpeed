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
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.inference_params import InferenceParams
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams

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
            position_ids_buffer = torch.empty((dynamic_seq_len, bsz), dtype=torch.int64, device=torch.cuda.current_device())
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
            position_ids_buffer = torch.empty((dynamic_seq_len, bsz), dtype=torch.int64, device=torch.cuda.current_device())
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
    
    if packed_seq_params is not None:
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
    if packed_seq_params is not None:
        core_attn_out = rearrange(core_attn_out, '(b s) h d -> s b (h d)', b=bsz)

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

        q_index, kv_index = compute_qkv_index(actual_seq_len.clone().tolist())
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
    t: Tensor, cu_seqlens: Tensor, freqs: Tensor, rotary_interleaved: bool = False, multi_latent_attention: bool = False, mscale: float = 1.0
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
    args = get_args()

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
        if transformer.input_tensor is not None:
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
