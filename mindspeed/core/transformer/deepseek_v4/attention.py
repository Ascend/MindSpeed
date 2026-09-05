# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# pylint: disable=too-many-lines

from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Union

import torch
from einops import rearrange

from megatron.core import parallel_state
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer import ModuleSpec, build_module
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.training import get_args

from mindspeed.core.context_parallel.deepseek_v4_context_parallel import (
    DeepSeekV4CPContextParallel,
    build_deepseek_v4_cp_packed_seq_metadata,
    build_deepseek_v4_causal_cmp_sparse_indices,
    compact_deepseek_v4_compressed_kv,
    exchange_deepseek_v4_packed_previous_window,
    exchange_deepseek_v4_previous_tails,
    exchange_deepseek_v4_previous_window,
    launch_deepseek_v4_allgather_compressed_kv,
    prepare_deepseek_v4_compression_candidates_for_cp,
    wait_deepseek_v4_compressed_kv,
    flatten_deepseek_v4_cp_tensor_to_tnd,
    build_deepseek_v4_owned_runtime_metadata,
)
from mindspeed.core.transformer.custom_layers.transformer_engine import PTNorm

from .compressor import get_compressor_spec
from .deepseek_utils import (
    apply_rotary_emb,
    build_deepseek_v4_cp_local_packed_position_ids,
    select_deepseek_v4_cp_packed_seq_params,
)
from mindspeed.core.context_parallel.deepseek_v4_context_parallel._utils import normalize_cu_seqlens
from .cp_indexer import (
    build_deepseek_v4_dense_indexer_compact_indices,
    finalize_deepseek_v4_fused_indexer_compact_indices,
    run_deepseek_v4_right_aligned_fused_indexer,
    should_use_deepseek_v4_cp_indexer_loss,
)
from .indexer import get_dsa_indexer_spec
from .indexer_loss import DSAIndexerLossLoggingHelper
from .linear import LinearNoTP


class LayerCompressMode(Enum):
    NO_COMPRESS = auto()
    COMPRESSOR_ONLY = auto()
    INDEXER = auto()


@dataclass
class DeepSeekV4SelfAttentionCPSubmodules(SelfAttentionSubmodules):
    linear_q: Union[ModuleSpec, type] = None
    linear_kv: Union[ModuleSpec, type] = None
    linear_o_down_proj: Union[ModuleSpec, type] = None
    linear_o_up_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    kv_layernorm: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    dsa_indexer: Union[ModuleSpec, type] = None
    compressor: Union[ModuleSpec, type] = None


def get_deepseek_v4_cp_self_attn_submodules(
    qk_layernorm,
    mla_mm_split,
    enable_dsa_indexer,
    use_te=False,
    compressor=True,
):
    del mla_mm_split
    if use_te:
        from megatron.core.transformer.custom_layers.transformer_engine import (
            TEColumnParallelLinear,
            TERowParallelLinear,
        )

        column_linear, row_linear = TEColumnParallelLinear, TERowParallelLinear
    else:
        column_linear, row_linear = ColumnParallelLinear, RowParallelLinear
    return DeepSeekV4SelfAttentionCPSubmodules(
        linear_q=LinearNoTP,
        linear_kv=LinearNoTP,
        linear_o_down_proj=column_linear,
        linear_o_up_proj=row_linear,
        q_layernorm=PTNorm if qk_layernorm else IdentityOp,
        kv_layernorm=PTNorm if qk_layernorm else IdentityOp,
        linear_q_up_proj=column_linear,
        dsa_indexer=get_dsa_indexer_spec(enable_dsa_indexer, compressor=compressor),
        compressor=get_compressor_spec() if compressor else None,
    )


def rms_norm_query_heads(query, eps, use_fused=False):
    if use_fused:
        import torch_npu

        gamma = torch.ones(query.shape[-1], device=query.device, dtype=torch.float32)
        return torch_npu.npu_rms_norm(query, gamma=gamma, epsilon=eps)[0]
    return query * torch.rsqrt(query.square().mean(-1, keepdim=True) + eps)


def _prepare_shared_fp32_candidate(candidate_blocks, valid_mask):
    candidate = candidate_blocks.float()
    if valid_mask is None:
        return candidate
    valid_mask = valid_mask.to(device=candidate.device, dtype=torch.bool)
    return candidate * valid_mask.reshape((-1,) + (1,) * (candidate.dim() - 1)).to(candidate.dtype)


def _gather_sequence_parallel_to_cp_local(tensor):
    """Restore the TP sequence shard without gathering across CP ranks."""
    return gather_from_sequence_parallel_region(
        tensor,
        group=parallel_state.get_tensor_model_parallel_group(),
    )


class _CPCompressor:
    """CP-aware adapter over ``Compressor``: left-context, RoPE positions, shared candidate."""

    def __init__(self, compressor, freqs_cis, packed, prepared_candidate=None):
        self.compressor = compressor
        self.freqs_cis = freqs_cis
        self.packed = packed
        self.prepared_candidate = prepared_candidate

    def compress(self, candidate_blocks, context):
        valid_mask = getattr(context, "valid_mask", None)
        if self.prepared_candidate is not None:
            candidate = self.prepared_candidate
            if valid_mask is not None:
                valid_mask = valid_mask.to(device=candidate.device, dtype=torch.bool)
        else:
            candidate = candidate_blocks
            if getattr(get_args(), "fp8", None) is None:
                candidate = candidate.float()
            if valid_mask is not None:
                valid_mask = valid_mask.to(device=candidate.device, dtype=torch.bool)
                candidate = candidate * valid_mask.reshape((-1,) + (1,) * (candidate.dim() - 1)).to(candidate.dtype)

        projected_kv, projected_score = self.compressor.project_candidate_blocks(candidate)
        if self.compressor.overlap:
            projected_kv, projected_score = self._add_left_context(
                projected_kv,
                projected_score,
                context,
                candidate_blocks.dtype,
            )
        positions = (
            getattr(context, "candidate_sample_positions", None)
            if self.packed
            else getattr(context, "candidate_positions", None)
        )
        if positions is None:
            positions = context.candidate_starts
        positions = positions.to(device=projected_kv.device, dtype=torch.long)
        max_position = int(positions.max().item()) if positions.numel() else 0
        if max_position >= self.freqs_cis.shape[0]:
            raise ValueError(
                "DeepSeek-V4 compressor candidate position exceeds RoPE table: "
                f"max={max_position}, length={self.freqs_cis.shape[0]}."
            )
        freqs = self.freqs_cis.to(projected_kv.device).index_select(0, positions)
        return self.compressor.compress_candidate_blocks(
            projected_kv,
            projected_score,
            freqs,
            candidate_blocks.dtype,
            valid_mask=valid_mask,
            batch_shared_sequence=bool(getattr(context, "batch_shared_sequence", False)),
        )

    def _add_left_context(self, projected_kv, projected_score, context, output_dtype):
        ratio = int(self.compressor.compress_ratio)
        dim = int(self.compressor.head_dim)
        left = getattr(context, "left_context_blocks", None)
        if left is None:
            left = getattr(context, "left_context_boundary_blocks", None)
        left_valid = getattr(context, "left_context_valid_mask", None)
        if left is None or left_valid is None:
            raise ValueError("C4A candidate compression requires left-context blocks.")
        if getattr(get_args(), "fp8", None) is None:
            if left.device != projected_kv.device or left.dtype != output_dtype:
                left = left.to(device=projected_kv.device, dtype=output_dtype)
            left = _as_float32(left)
        elif left.device != projected_kv.device or left.dtype != output_dtype:
            left = left.to(device=projected_kv.device, dtype=output_dtype)
        left_kv, left_score = self.compressor.project_candidate_blocks(left)
        shape = projected_kv.shape[:1] + (2 * ratio,) + projected_kv.shape[2:-1] + (dim,)
        overlap_kv = projected_kv.new_zeros(shape)
        overlap_score = projected_score.new_full(shape, float("-inf"))
        overlap_kv[:, ratio:] = projected_kv[..., dim:]
        overlap_score[:, ratio:] = projected_score[..., dim:]
        valid = left_valid.to(device=projected_kv.device, dtype=torch.bool)
        candidate_valid = getattr(context, "valid_mask", None)
        if candidate_valid is not None:
            valid = valid & candidate_valid.to(device=projected_kv.device, dtype=torch.bool)

        source_indices = getattr(context, "left_context_source_indices", None)
        boundary_indices = getattr(context, "left_context_boundary_indices", None)
        if source_indices is not None and boundary_indices is not None:
            source_indices = source_indices.to(device=projected_kv.device, dtype=torch.long)
            boundary_indices = boundary_indices.to(device=projected_kv.device, dtype=torch.long)
            if source_indices.numel() != projected_kv.shape[0]:
                raise ValueError(
                    "left_context_source_indices must match the candidate count: "
                    f"expected {projected_kv.shape[0]}, got {source_indices.numel()}."
                )

            local_targets = torch.nonzero(
                valid & (source_indices >= 0),
                as_tuple=False,
            ).flatten()
            if local_targets.numel():
                local_sources = source_indices.index_select(0, local_targets)
                overlap_kv[local_targets, :ratio] = projected_kv[local_sources, ..., :dim]
                overlap_score[local_targets, :ratio] = projected_score[local_sources, ..., :dim]

            boundary_valid = valid.index_select(0, boundary_indices)
            boundary_targets = boundary_indices[boundary_valid]
            if boundary_targets.numel():
                if left_kv.shape[0] == source_indices.shape[0]:
                    boundary_rows = boundary_targets
                elif left_kv.shape[0] == boundary_indices.shape[0]:
                    boundary_rows = torch.nonzero(boundary_valid, as_tuple=False).flatten()
                else:
                    raise ValueError(
                        "left-context projection count must match either the candidate count "
                        "or the boundary count: "
                        f"candidates={source_indices.shape[0]}, boundaries={boundary_indices.shape[0]}, "
                        f"projection={left_kv.shape[0]}."
                    )
                overlap_kv[boundary_targets, :ratio] = left_kv[boundary_rows, ..., :dim]
                overlap_score[boundary_targets, :ratio] = left_score[boundary_rows, ..., :dim]
        elif torch.any(valid):
            if left_kv.shape[0] != valid.shape[0]:
                raise ValueError(
                    "left-context projection count must match the candidate count when "
                    "reuse metadata is unavailable: "
                    f"candidates={valid.shape[0]}, projection={left_kv.shape[0]}."
                )
            overlap_kv[valid, :ratio] = left_kv[valid, ..., :dim]
            overlap_score[valid, :ratio] = left_score[valid, ..., :dim]
        return overlap_kv, overlap_score


@dataclass(frozen=True)
class _CPRuntime:
    layout: str
    query: torch.Tensor
    ori_kv: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_ori_kv: torch.Tensor
    cu_seqlens_global: torch.Tensor
    query_positions: torch.Tensor
    local_seq_offset: int
    cp_group: object
    cp_global_ranks: object
    cp_size: int
    cp_rank: int
    batch_size: int
    seq_len: int
    window_size: int

    @property
    def seq_dim(self):
        return 1 if self.layout == "BSND" else 0

    @property
    def is_bsnd(self):
        return self.layout == "BSND"

    def shared_query_positions(self):
        if not self.is_bsnd:
            return self.query_positions
        positions = self.query_positions.reshape(self.batch_size, self.seq_len)
        sample_starts = self.cu_seqlens_global[:-1].to(
            device=positions.device,
            dtype=torch.long,
        )
        return (positions - sample_starts.unsqueeze(1))[0].contiguous()

    def shared_cu_seqlens_global(self):
        if not self.is_bsnd:
            return self.cu_seqlens_global
        if self.cu_seqlens_global.numel() < 2:
            zero = self.cu_seqlens_global.new_zeros(1)
            return torch.cat((zero, zero))
        zero = self.cu_seqlens_global.new_zeros(1)
        length = (self.cu_seqlens_global[1] - self.cu_seqlens_global[0]).reshape(1)
        return torch.cat((zero, length))


def _model_tensor_to_cp_layout(tensor, layout, *, shared_kv=False):
    if layout == "TND":
        return flatten_deepseek_v4_cp_tensor_to_tnd(
            tensor,
            seq_dim=0,
            batch_dim=1,
            shared_kv=shared_kv,
        )
    if layout != "BSND":
        raise ValueError(f"Unsupported DeepSeek V4 CP layout: {layout}.")
    if tensor.dim() < 2:
        raise ValueError("DeepSeek V4 CP tensors must contain sequence and batch dimensions.")
    result = tensor.transpose(0, 1).contiguous()
    if shared_kv and result.dim() == 3:
        result = result.unsqueeze(2)
    return result


def _build_runtime_sequence_metadata(
    query,
    ori_kv,
    seq_len,
    batch_size,
    cp_size,
    cp_rank,
    tp_size,
    tp_rank,
    sequence_parallel,
    device,
    packed_seq_params=None,
    layout="BSND",
):
    if layout == "TND":
        query_token_count = int(query.shape[0])
        kv_token_count = int(ori_kv.shape[0])
        if batch_size != 1:
            raise NotImplementedError("DeepSeek V4 packed CP currently requires batch_size=1.")
        if kv_token_count % int(batch_size) != 0:
            raise ValueError("DeepSeek V4 packed KV token count must be divisible by batch size.")
        cp_local_kv_seq_len = kv_token_count // int(batch_size)
    elif layout == "BSND":
        if query.dim() != 4 or query.shape[0] != batch_size or query.shape[1] != seq_len:
            raise ValueError(
                "DeepSeek V4 CP BSND query must have shape [B, S, N, D]: "
                f"expected batch={batch_size}, seq={seq_len}, got={tuple(query.shape)}."
            )
        if ori_kv.dim() != 4 or ori_kv.shape[0] != batch_size or ori_kv.shape[2] != 1:
            raise ValueError(
                "DeepSeek V4 CP BSND KV must have shape [B, S, 1, D]: "
                f"expected batch={batch_size}, got={tuple(ori_kv.shape)}."
            )
        query_token_count = int(batch_size) * int(seq_len)
        cp_local_kv_seq_len = int(ori_kv.shape[1])
        kv_token_count = int(batch_size) * cp_local_kv_seq_len
    else:
        raise ValueError(f"Unsupported DeepSeek V4 CP layout: {layout}.")

    if query_token_count != int(seq_len) * int(batch_size):
        raise ValueError(
            "DeepSeek V4 CP query token count does not match the model layout: "
            f"expected={int(seq_len) * int(batch_size)}, got={query_token_count}."
        )

    tp_size = int(tp_size)
    tp_rank = int(tp_rank)
    if tp_size <= 0 or tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError(f"Invalid tensor-parallel topology: tp_size={tp_size}, tp_rank={tp_rank}.")

    if sequence_parallel:
        complete_tp_seq_len = int(seq_len) * tp_size
        if cp_local_kv_seq_len == int(seq_len):
            query_tp_offset = 0
        elif cp_local_kv_seq_len == complete_tp_seq_len:
            query_tp_offset = tp_rank * int(seq_len)
        else:
            raise ValueError(
                "DeepSeek V4 CP sequence-parallel Query/KV layout is inconsistent: "
                f"expected KV length {seq_len} or {complete_tp_seq_len}, "
                f"got={cp_local_kv_seq_len}."
            )
    else:
        if cp_local_kv_seq_len != int(seq_len):
            raise ValueError(
                "DeepSeek V4 CP without sequence parallelism requires matching query and KV lengths: "
                f"query={seq_len}, kv={cp_local_kv_seq_len}."
            )
        query_tp_offset = 0

    kv_local_seq_offset = int(cp_rank) * cp_local_kv_seq_len
    query_local_seq_offset = kv_local_seq_offset + query_tp_offset

    if packed_seq_params is not None:
        if layout != "TND":
            raise ValueError("DeepSeek V4 packed CP sequences must use TND layout.")
        packed_cu = normalize_cu_seqlens(
            packed_seq_params.cu_seqlens_q,
            device,
        )
        global_total = int(packed_cu[-1].item())
        expected_global_total = kv_token_count * int(cp_size)
        if global_total != expected_global_total:
            raise ValueError(
                "DeepSeek V4 packed cu_seqlens_q total must equal local KV token count * CP size: "
                f"cu_total={global_total}, local_kv_tokens={kv_token_count}, cp_size={cp_size}."
            )
        query_metadata = build_deepseek_v4_cp_packed_seq_metadata(
            packed_cu,
            local_seq_offset=query_local_seq_offset,
            local_seq_len=query_token_count,
            device=device,
        )
        kv_metadata = build_deepseek_v4_cp_packed_seq_metadata(
            packed_cu,
            local_seq_offset=kv_local_seq_offset,
            local_seq_len=kv_token_count,
            device=device,
        )
        return (
            query_metadata.cu_seqlens_q,
            kv_metadata.cu_seqlens_ori_kv,
            query_metadata.cu_seqlens,
            query_metadata.query_positions,
            kv_metadata.local_seq_offset,
        )

    cu_q = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * int(seq_len)
    cu_kv = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * int(cp_local_kv_seq_len)
    global_sample_len = cp_local_kv_seq_len * int(cp_size)
    cu_global = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * global_sample_len
    local_positions = torch.arange(
        query_local_seq_offset,
        query_local_seq_offset + int(seq_len),
        dtype=torch.long,
        device=device,
    )
    sample_offsets = torch.arange(batch_size, dtype=torch.long, device=device) * global_sample_len
    query_positions = (local_positions.unsqueeze(0) + sample_offsets.unsqueeze(1)).reshape(-1)
    return cu_q, cu_kv, cu_global, query_positions, kv_local_seq_offset


def _build_runtime(query, ori_kv, packed_seq_params, config, window_size, mtp_idx=0):
    del mtp_idx
    layout = "TND" if packed_seq_params is not None else "BSND"
    query_cp = _model_tensor_to_cp_layout(query, layout)
    ori_kv_cp = _model_tensor_to_cp_layout(ori_kv, layout, shared_kv=True)
    batch_size = int(query.shape[1])
    seq_len = int(query.shape[0])
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    (
        cu_q,
        cu_kv,
        global_cu,
        positions,
        local_offset,
    ) = _build_runtime_sequence_metadata(
        query_cp,
        ori_kv_cp,
        seq_len,
        batch_size,
        cp_size,
        cp_rank,
        tp_size,
        tp_rank,
        bool(getattr(config, "sequence_parallel", False)),
        query.device,
        packed_seq_params=packed_seq_params,
        layout=layout,
    )
    return _CPRuntime(
        layout=layout,
        query=query_cp,
        ori_kv=ori_kv_cp,
        cu_seqlens_q=cu_q,
        cu_seqlens_ori_kv=cu_kv,
        cu_seqlens_global=global_cu,
        query_positions=positions,
        local_seq_offset=local_offset,
        cp_group=parallel_state.get_context_parallel_group(),
        cp_global_ranks=parallel_state.get_context_parallel_global_ranks(),
        cp_size=cp_size,
        cp_rank=cp_rank,
        batch_size=batch_size,
        seq_len=seq_len,
        window_size=max(int(window_size), 1),
    )


def _as_float32(tensor):
    if tensor is None or tensor.dtype == torch.float32:
        return tensor
    return tensor.float()


def _adjust_cu_seqlens_for_window(cu_seqlens, tensor, layout, extra_tokens=None):
    if layout == "BSND":
        return torch.arange(
            int(tensor.shape[0]) + 1,
            dtype=torch.int32,
            device=tensor.device,
        ) * int(tensor.shape[1])
    actual = int(tensor.shape[0])
    cu = cu_seqlens.to(device=tensor.device, dtype=torch.int32)
    if extra_tokens is None:
        extra = actual - int(cu[-1].item())
    else:
        extra = int(extra_tokens)
    if extra <= 0:
        return cu
    if cu.numel() == 2:
        return torch.tensor([0, actual], dtype=torch.int32, device=tensor.device)
    lengths = torch.diff(cu).clone()
    lengths[0] += extra
    return torch.cat([cu.new_zeros(1), torch.cumsum(lengths, 0, dtype=torch.int32)])


class DeepSeekV4SelfAttentionCP(MegatronModule):
    """MindSpeed-owned DeepSeek-V4 attention for ``deepseek_v4_cp_algo``."""

    enable_compression = True

    def __init__(
        self,
        config,
        submodules: DeepSeekV4SelfAttentionCPSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.causal,
        cp_comm_type=None,
    ):
        del cp_comm_type
        super().__init__(config=config)
        args = get_args()
        self.head_dim = int(args.qk_head_dim)
        self.rope_head_dim = int(args.qk_pos_emb_head_dim)
        self.q_lora_rank = int(args.q_lora_rank)
        self.o_lora_rank = int(args.o_lora_rank)
        self.world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.n_groups = int(args.o_groups)
        self.n_local_groups = self.n_groups // self.world_size
        self.dim = int(args.hidden_size)
        self.layer_number = layer_number + get_transformer_layer_offset(config)
        self.mtp_idx = 0
        self.n_heads = int(args.num_attention_heads)
        self.n_local_heads = self.n_heads // self.world_size
        self.attn_sink = torch.nn.Parameter(torch.zeros(self.n_local_heads, dtype=torch.float32))

        self.linear_q = build_module(
            submodules.linear_q,
            self.dim,
            self.q_lora_rank,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=config.add_bias_linear or config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="q",
        )
        self.linear_kv = build_module(
            submodules.linear_kv,
            self.dim,
            self.head_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=config.add_bias_linear or config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv",
        )
        self.q_layernorm = build_module(
            submodules.q_layernorm,
            hidden_size=self.q_lora_rank,
            config=config,
            eps=config.layernorm_epsilon,
        )
        self.kv_layernorm = build_module(
            submodules.kv_layernorm,
            hidden_size=self.head_dim,
            config=config,
            eps=config.layernorm_epsilon,
        )
        self.linear_q_up_proj = build_module(
            submodules.linear_q_up_proj,
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=config.add_bias_linear or config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="q_up",
        )
        self.linear_o_down_proj = build_module(
            submodules.linear_o_down_proj,
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=config.add_bias_linear or config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="o_down",
        )
        self.linear_o_up_proj = build_module(
            submodules.linear_o_up_proj,
            self.n_groups * self.o_lora_rank,
            self.dim,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="o_up_proj",
        )
        self.compressor = None
        self.indexer = None
        self.compress_ratio = 1
        if not self.enable_compression:
            self.mode = LayerCompressMode.NO_COMPRESS
        else:
            self.compress_ratio = int(args.compress_ratios[self.layer_number - 1])
        if self.compress_ratio <= 1:
            self.compress_ratio = 1
            self.mode = LayerCompressMode.NO_COMPRESS
        elif self.compress_ratio == 4:
            self.mode = LayerCompressMode.INDEXER
        else:
            self.mode = LayerCompressMode.COMPRESSOR_ONLY
        self.softmax_scale = self.head_dim**-0.5
        self.window_size = int(getattr(args, "sliding_window_size", None) or 128)
        if self.enable_compression and self.mode != LayerCompressMode.NO_COMPRESS:
            self.compressor = build_module(
                submodules.compressor,
                config=config,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
            )
            if self.mode == LayerCompressMode.INDEXER and getattr(args, "enable_dsa_indexer", False):
                self.indexer = build_module(
                    submodules.dsa_indexer,
                    config=config,
                    layer_number=self.layer_number,
                )

    def _get_position_freqs(self, rotary_pos_emb, q_len, batch_size, packed, packed_seq_params=None):
        base = rotary_pos_emb[0] if self.mode != LayerCompressMode.NO_COMPRESS else rotary_pos_emb[1]
        if packed:
            if packed_seq_params is None:
                raise ValueError("packed_seq_params is required for packed DeepSeek V4 CP attention.")
            q_positions = build_deepseek_v4_cp_local_packed_position_ids(
                packed_seq_params,
                q_len,
                parallel_state.get_context_parallel_world_size(),
                parallel_state.get_context_parallel_rank(),
                base.device,
                position_count=q_len
                * (parallel_state.get_tensor_model_parallel_world_size() if self.config.sequence_parallel else 1),
                tp_size=parallel_state.get_tensor_model_parallel_world_size(),
                tp_rank=parallel_state.get_tensor_model_parallel_rank(),
                sequence_parallel=self.config.sequence_parallel,
                get_global=True,
            )
            kv_positions = build_deepseek_v4_cp_local_packed_position_ids(
                packed_seq_params,
                q_len,
                parallel_state.get_context_parallel_world_size(),
                parallel_state.get_context_parallel_rank(),
                base.device,
                position_count=q_len,
                tp_size=parallel_state.get_tensor_model_parallel_world_size(),
                tp_rank=parallel_state.get_tensor_model_parallel_rank(),
                sequence_parallel=self.config.sequence_parallel,
                get_global=False,
            )
            return q_positions, kv_positions, base
        cp_rank = parallel_state.get_context_parallel_rank()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        total = q_len * (tp_size if self.config.sequence_parallel else 1)
        start = cp_rank * total
        q_positions = torch.arange(start, start + total, device=base.device, dtype=torch.long)
        kv_start = start + (
            parallel_state.get_tensor_model_parallel_rank() * q_len if self.config.sequence_parallel else 0
        )
        kv_positions = torch.arange(kv_start, kv_start + q_len, device=base.device, dtype=torch.long)
        return q_positions, kv_positions, base

    def _apply_qk_rope(self, q, kv, q_freqs, kv_freqs):
        q = q.transpose(0, 1)
        kv = kv.transpose(0, 1)
        q[..., -self.rope_head_dim :] = apply_rotary_emb(q[..., -self.rope_head_dim :], q_freqs)
        kv[..., -self.rope_head_dim :] = apply_rotary_emb(kv[..., -self.rope_head_dim :], kv_freqs)
        return q.transpose(0, 1), kv.transpose(0, 1)

    def _prepare_compressed_kv(self, runtime, source, freqs, ratio):
        previous_ori = None
        previous_source = None
        if ratio > 1:
            previous_ori, previous_source = exchange_deepseek_v4_previous_tails(
                local_tensors=(runtime.ori_kv, source),
                tail_lengths=(runtime.window_size, ratio * 2 if ratio == 4 else ratio),
                cp_group=runtime.cp_group,
                cp_global_ranks=runtime.cp_global_ranks,
                seq_dim=runtime.seq_dim,
            )
        candidates = prepare_deepseek_v4_compression_candidates_for_cp(
            source,
            ratio,
            cp_group=runtime.cp_group,
            cp_global_ranks=runtime.cp_global_ranks,
            seq_dim=runtime.seq_dim,
            cu_seqlens=runtime.cu_seqlens_global,
            local_seq_offset=runtime.local_seq_offset,
            include_left_context=ratio == 4,
            previous_tail=previous_source,
            use_compact_candidate_view=getattr(get_args(), "fp8", None) is None,
            batch_shared_sequence=runtime.is_bsnd,
        )
        shared_candidate = None
        if (
            self.indexer is not None
            and getattr(self.indexer, "kv_compressor", None) is not None
            and getattr(get_args(), "fp8", None) is None
        ):
            shared_candidate = _prepare_shared_fp32_candidate(
                candidates.candidate_blocks,
                getattr(candidates.compress_context, "valid_mask", None),
            )
        local_compressed = _CPCompressor(
            self.compressor,
            freqs,
            not runtime.is_bsnd,
            prepared_candidate=shared_candidate,
        ).compress(candidates.candidate_blocks, candidates.compress_context)
        pending = launch_deepseek_v4_allgather_compressed_kv(candidates, local_compressed)
        indexer_pending = None
        if self.indexer is not None and getattr(self.indexer, "kv_compressor", None) is not None:
            indexer_candidates = replace(
                candidates,
                candidate_blocks=candidates.candidate_blocks.detach(),
                compress_context=replace(
                    candidates.compress_context,
                    left_context_blocks=(
                        None
                        if candidates.compress_context.left_context_blocks is None
                        else candidates.compress_context.left_context_blocks.detach()
                    ),
                    left_context_boundary_blocks=(
                        None
                        if candidates.compress_context.left_context_boundary_blocks is None
                        else candidates.compress_context.left_context_boundary_blocks.detach()
                    ),
                ),
            )
            indexer_compressed = _CPCompressor(
                self.indexer.kv_compressor,
                freqs,
                not runtime.is_bsnd,
                prepared_candidate=None if shared_candidate is None else shared_candidate.detach(),
            ).compress(
                indexer_candidates.candidate_blocks,
                indexer_candidates.compress_context,
            )
            indexer_pending = launch_deepseek_v4_allgather_compressed_kv(
                indexer_candidates,
                indexer_compressed,
            )
        return (
            previous_ori,
            previous_source,
            candidates,
            pending,
            indexer_pending,
        )

    def _windowed_ori_kv(self, runtime, previous_ori, packed):
        if packed:
            return exchange_deepseek_v4_packed_previous_window(
                local_tensor=runtime.ori_kv,
                window_size=runtime.window_size,
                cu_seqlens=runtime.cu_seqlens_global,
                local_seq_offset=runtime.local_seq_offset,
                cp_group=runtime.cp_group,
                cp_global_ranks=runtime.cp_global_ranks,
                seq_dim=0,
                previous_tail=previous_ori,
            )
        windowed = exchange_deepseek_v4_previous_window(
            local_tensor=runtime.ori_kv,
            window_size=runtime.window_size,
            cp_group=runtime.cp_group,
            cp_global_ranks=runtime.cp_global_ranks,
            seq_dim=runtime.seq_dim,
            previous_tail=previous_ori,
        )
        extra_tokens = int(windowed.shape[runtime.seq_dim]) - int(runtime.ori_kv.shape[runtime.seq_dim])
        return windowed, _adjust_cu_seqlens_for_window(
            runtime.cu_seqlens_ori_kv,
            windowed,
            runtime.layout,
            extra_tokens=extra_tokens,
        )

    def _linear_o_down(self, grouped_output):
        weight = rearrange(
            self.linear_o_down_proj.weight,
            "(g l) (d h) -> g l (d h)",
            d=self.head_dim // self.n_groups,
            l=self.o_lora_rank,
            h=self.n_heads,
            g=self.n_local_groups,
        )
        output = torch.einsum("sbgd,gld->sbgl", grouped_output, weight)

        # Keep the output projection compatible with the framework LoRA
        # wrapper without importing any framework-specific implementation.
        linear = self.linear_o_down_proj
        if not (hasattr(linear, "lora_A") and hasattr(linear, "lora_B")):
            return output
        if getattr(linear, "disable_adapters", False) or getattr(linear, "merged", False):
            return output

        active_adapters = getattr(linear, "active_adapters", None)
        if active_adapters is None:
            active_adapter = getattr(linear, "active_adapter", None)
            active_adapters = [active_adapter] if isinstance(active_adapter, str) else active_adapter
        if not active_adapters:
            return output

        for active_adapter in active_adapters:
            if active_adapter not in linear.lora_A.keys() or active_adapter not in linear.lora_B.keys():
                continue

            lora_a = linear.lora_A[active_adapter].weight
            lora_b = linear.lora_B[active_adapter].weight
            scaling = linear.scaling[active_adapter]
            lora_input = grouped_output.to(lora_a.dtype)
            lora_a_output = torch.einsum("sbgd,rd->sbgr", lora_input, lora_a)
            lora_b_weight = rearrange(
                lora_b,
                "(g l) r -> g l r",
                g=self.n_local_groups,
                l=self.o_lora_rank,
            )
            lora_delta = torch.einsum("sbgr,glr->sbgl", lora_a_output, lora_b_weight) * scaling
            output = output + lora_delta.to(output.dtype)

        return output

    def _run_cp_attention(
        self,
        q,
        kv,
        hidden_for_compress,
        q_compressed,
        global_freqs,
        local_freqs,
        packed_seq_params,
    ):
        args = get_args()
        runtime = _build_runtime(
            q,
            kv,
            packed_seq_params,
            self.config,
            max(self.window_size, self.compress_ratio),
            mtp_idx=self.mtp_idx,
        )
        ratio = self.compress_ratio
        ori_windowed = None
        prepared_kv = None
        compacted_kv = None
        cmp = None
        cmp_is_causal = False
        block_starts = None
        query_index = key_index = weights = None
        previous_ori = previous_source = None
        if ratio > 1:
            (
                previous_ori,
                previous_source,
                candidates,
                pending,
                indexer_pending,
            ) = self._prepare_compressed_kv(
                runtime,
                (
                    hidden_for_compress.transpose(0, 1).contiguous()
                    if runtime.is_bsnd
                    else flatten_deepseek_v4_cp_tensor_to_tnd(
                        hidden_for_compress,
                        seq_dim=0,
                        batch_dim=1,
                    )
                ),
                global_freqs if packed_seq_params is None else global_freqs,
                ratio,
            )
            prepared_kv = wait_deepseek_v4_compressed_kv(pending)
            compacted_kv, block_starts = compact_deepseek_v4_compressed_kv(
                prepared_kv,
                seq_dim=runtime.seq_dim,
            )
            ori_windowed, cu_ori_windowed = self._windowed_ori_kv(
                runtime,
                previous_ori,
                packed_seq_params is not None,
            )
            if self.indexer is not None:
                if indexer_pending is None:
                    raise RuntimeError("C4A requires an indexer compressed-key path.")
                indexer_key, _ = compact_deepseek_v4_compressed_kv(
                    wait_deepseek_v4_compressed_kv(indexer_pending),
                    seq_dim=runtime.seq_dim,
                )
                query_index, weights = self.indexer.forward_query_weights(
                    hidden_for_compress.detach(),
                    q_compressed.detach(),
                    local_freqs,
                )
                if runtime.is_bsnd:
                    key_index = indexer_key
                elif indexer_key.dim() == 4:
                    key_index = flatten_deepseek_v4_cp_tensor_to_tnd(
                        indexer_key,
                        seq_dim=0,
                        batch_dim=1,
                    )
                else:
                    key_index = indexer_key

                use_fused_indexer = bool(getattr(args, "use_fused_lightning_indexer", False))
                if use_fused_indexer:
                    with torch.no_grad():
                        cmp = run_deepseek_v4_right_aligned_fused_indexer(
                            self.indexer,
                            hidden_for_compress,
                            query_index,
                            key_index,
                            weights,
                            (runtime.shared_query_positions() if runtime.is_bsnd else runtime.query_positions),
                            runtime.cu_seqlens_global,
                            ratio,
                            self.indexer.index_topk,
                            0,
                            identity_single_sample=(
                                packed_seq_params is None
                                and bool(getattr(prepared_kv.metadata, "is_identity_compact_order", False))
                                and (runtime.batch_size == 1 or runtime.is_bsnd)
                            ),
                            layout=runtime.layout,
                        )
                    cmp, cmp_is_causal = finalize_deepseek_v4_fused_indexer_compact_indices(
                        cmp,
                        block_starts,
                        ratio,
                        runtime.shared_cu_seqlens_global() if runtime.is_bsnd else runtime.cu_seqlens_global,
                        runtime.shared_query_positions() if runtime.is_bsnd else runtime.query_positions,
                        identity_compact_order=(
                            packed_seq_params is None
                            and bool(getattr(prepared_kv.metadata, "is_identity_compact_order", False))
                            and (runtime.batch_size == 1 or runtime.is_bsnd)
                        ),
                        layout=runtime.layout,
                        batch_size=runtime.batch_size,
                        seq_len=runtime.seq_len,
                    )
                else:
                    cmp = build_deepseek_v4_dense_indexer_compact_indices(
                        query_index,
                        key_index,
                        weights,
                        runtime.query_positions,
                        block_starts,
                        ratio,
                        runtime.cu_seqlens_global,
                        self.indexer.index_topk,
                        layout=runtime.layout,
                    )
            elif ratio == 4:
                valid_mask = torch.ones_like(block_starts, dtype=torch.bool)
                cmp = build_deepseek_v4_causal_cmp_sparse_indices(
                    runtime.query_positions,
                    block_starts,
                    valid_mask,
                    ratio,
                    cu_seqlens=runtime.cu_seqlens_global,
                    sparse_count=int(getattr(args, "index_topk", 512)),
                )
                if runtime.is_bsnd:
                    cmp = cmp.reshape(runtime.batch_size, runtime.seq_len, -1)
            else:
                cmp = None
        else:
            ori_windowed, cu_ori_windowed = self._windowed_ori_kv(
                runtime,
                None,
                packed_seq_params is not None,
            )

        use_loss = self.indexer is not None and should_use_deepseek_v4_cp_indexer_loss(
            self,
            args,
            float(getattr(args, "indexer_loss_coeff", 0.0)),
        )
        if use_loss:
            if runtime.is_bsnd:
                query_index = query_index.transpose(0, 1).contiguous()
                weights = _as_float32(weights.transpose(0, 1).contiguous())
            else:
                query_index = flatten_deepseek_v4_cp_tensor_to_tnd(
                    query_index,
                    seq_dim=0,
                    batch_dim=1,
                )
                weights = _as_float32(
                    flatten_deepseek_v4_cp_tensor_to_tnd(
                        weights,
                        seq_dim=0,
                        batch_dim=1,
                    )
                )
        owned_runtime = build_deepseek_v4_owned_runtime_metadata(
            query=runtime.query,
            layout=runtime.layout,
            cu_seqlens_q=runtime.cu_seqlens_q,
            cu_seqlens_ori_kv=cu_ori_windowed,
            cu_seqlens_global=runtime.cu_seqlens_global,
            query_positions=runtime.query_positions,
            local_seq_offset=runtime.local_seq_offset,
            batch_size=runtime.batch_size if runtime.is_bsnd else 1,
            compression_ratio=ratio,
        )
        cp = DeepSeekV4CPContextParallel(
            compression_ratio=ratio,
            layout_q=runtime.layout,
            layout_kv=runtime.layout,
        )
        output = cp.forward(
            q=runtime.query,
            ori_kv=ori_windowed,
            runtime_metadata=owned_runtime,
            prepared_compressed_kv=prepared_kv,
            sinks=_as_float32(self.attn_sink),
            softmax_scale=self.softmax_scale,
            cmp_sparse_indices=cmp,
            cmp_sparse_indices_are_causal=cmp_is_causal,
            query_index=query_index if use_loss else None,
            key_index=key_index if use_loss else None,
            weights=_as_float32(weights) if use_loss else None,
            loss_tracker=self.indexer_loss_tracker if use_loss else None,
            loss_coeff=getattr(args, "indexer_loss_coeff", 0.0),
            compacted_compressed_kv=compacted_kv,
            compacted_block_starts=block_starts,
        )
        if runtime.is_bsnd:
            return output.transpose(0, 1).contiguous()
        return output.reshape(q.shape[1], q.shape[0], q.shape[2], q.shape[3]).transpose(0, 1).contiguous()

    def indexer_loss_tracker(self, loss):
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss,
            self.layer_number,
            self.config.num_layers,
            avg_group=parallel_state.get_tensor_and_context_parallel_group(),
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb=None,
        start_pos=0,
        attention_bias=None,
        packed_seq_params=None,
        inference_context=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        sequence_len_offset=None,
    ):
        del attention_mask, attention_bias, rotary_pos_cos, rotary_pos_sin, sequence_len_offset
        if start_pos != 0:
            raise NotImplementedError("deepseek_v4_cp_algo currently supports training start_pos=0 only.")
        if inference_context is not None:
            raise NotImplementedError("deepseek_v4_cp_algo does not support inference KV cache.")
        if rotary_pos_emb is None:
            raise ValueError("DeepSeek V4 CP attention requires rotary_pos_emb.")
        if packed_seq_params is not None:
            packed_seq_params = select_deepseek_v4_cp_packed_seq_params(
                packed_seq_params,
                mtp_idx=self.mtp_idx,
            )
        q_len_local, batch_size, _ = hidden_states.shape
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        q_len = q_len_local * tp_size if self.config.sequence_parallel else q_len_local
        q_pos_ids, kv_pos_ids, freq_table = self._get_position_freqs(
            rotary_pos_emb,
            q_len_local,
            batch_size,
            packed_seq_params is not None,
            packed_seq_params=packed_seq_params,
        )
        freq_q = freq_table.index_select(0, q_pos_ids)
        freq_kv = freq_table.index_select(0, kv_pos_ids)

        q_compressed = self.linear_q(hidden_states)
        kv_compressed = self.linear_kv(hidden_states)
        q_compressed = self.q_layernorm(q_compressed)
        q_projected = self.linear_q_up_proj(q_compressed)
        q = q_projected[0] if isinstance(q_projected, (tuple, list)) else q_projected
        if tuple(q.shape) != (q_len, batch_size, self.n_local_heads * self.head_dim):
            raise ValueError(
                "DeepSeek V4 CP query projection must contain the complete local sequence: "
                f"expected={(q_len, batch_size, self.n_local_heads * self.head_dim)}, got={tuple(q.shape)}."
            )
        q = q.view(q_len, batch_size, self.n_local_heads, self.head_dim)
        q = rms_norm_query_heads(
            q,
            self.config.layernorm_epsilon,
            use_fused=bool(getattr(get_args(), "use_fused_rmsnorm", False)),
        )
        kv = self.kv_layernorm(kv_compressed).view(q_len_local, batch_size, self.head_dim)
        if packed_seq_params is not None:
            if freq_q.numel() != q.shape[0] * freq_q.shape[-1]:
                raise ValueError("Packed DeepSeek V4 RoPE metadata does not match query length.")
        q, kv = self._apply_qk_rope(q, kv, freq_q, freq_kv)

        hidden_for_cp = hidden_states
        if self.config.sequence_parallel:
            kv = _gather_sequence_parallel_to_cp_local(kv)
            hidden_for_cp = _gather_sequence_parallel_to_cp_local(hidden_for_cp)
            q_compressed = _gather_sequence_parallel_to_cp_local(q_compressed)

        global_freqs = freq_table
        local_freqs = freq_q
        output = self._run_cp_attention(
            q,
            kv,
            hidden_for_cp,
            q_compressed,
            global_freqs,
            local_freqs,
            packed_seq_params,
        )

        output = output.transpose(0, 1)
        output_rope = apply_rotary_emb(
            output[..., -self.rope_head_dim :],
            freq_q,
            inverse=True,
        )
        output = torch.cat((output[..., : -self.rope_head_dim], output_rope), dim=-1)
        output = output.transpose(0, 1).contiguous()
        output = rearrange(
            output,
            "s b (g h) d -> s b g (h d)",
            s=q_len,
            b=batch_size,
            g=self.n_groups // self.world_size,
            h=self.n_heads // self.n_groups,
            d=self.head_dim,
        )
        output = self._linear_o_down(output)
        output_projected = self.linear_o_up_proj(output.flatten(2))
        if isinstance(output_projected, (tuple, list)):
            return output_projected
        return output_projected, None


class DeepSeekV4MTPSelfAttentionCP(DeepSeekV4SelfAttentionCP):
    """MTP variant: it uses the same CP mechanics without DSA compression."""

    enable_compression = False

    def __init__(self, config, submodules, layer_number, attn_mask_type=AttnMaskType.causal, cp_comm_type=None):
        super().__init__(config, submodules, layer_number, attn_mask_type, cp_comm_type)
        self.is_mtp_attention = True
        self.indexer = None
        self.compressor = None
        self.compress_ratio = 1
        self.mode = LayerCompressMode.NO_COMPRESS


__all__ = [
    "DeepSeekV4MTPSelfAttentionCP",
    "DeepSeekV4SelfAttentionCP",
    "DeepSeekV4SelfAttentionCPSubmodules",
    "LayerCompressMode",
    "get_deepseek_v4_cp_self_attn_submodules",
]
