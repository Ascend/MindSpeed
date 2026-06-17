# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.jit import jit_fuser
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_context_parallel import (
    _all_to_all_cp2hp,
    _all_to_all_hp2cp,
    _redo_attention_load_balancing,
    _undo_attention_load_balancing,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push
from megatron.core.ssm.gated_delta_net import GatedDeltaNetSubmodules, _split_tensor_factory
from mindspeed.args_utils import get_full_args as get_args

try:
    from fla.modules.l2norm import l2norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    HAVE_FLA = True
except ImportError:
    chunk_gated_delta_rule = None

    HAVE_FLA = False

try:
    import fla_npu
    from fla.modules.convolution import causal_conv1d
except ImportError:
    causal_conv1d = None


def naive_l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    original_dtype = x.dtype
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    # Counteract verl's autocast promotion (bf16 -> fp32) by restoring original dtype
    return (x * inv_norm).to(original_dtype)


class GatedDeltaNet(MegatronModule):
    """Gated Delta Net (GDN) layer class

    GDN layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: GatedDeltaNetSubmodules,
        layer_number: int = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: Optional[float] = None,
        use_qk_l2norm: bool = True,
        A_init_range: Tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            config: The config of the model.
            submodules: Contains the module specs for the input and output linear layers.
            layer_number: The layer number of this GDN layer.
            bias: Whether to use bias in the linear layers.
            conv_bias: Whether to use bias in the causal convolution.
            conv_init: The initialization range for the causal convolution weights.
            use_qk_l2norm: Whether to use L2 normalization in the kernel of the gated delta rule.
            A_init_range: The initialization range for the attention weights.
            pg_collection: The required process groups to use for tensor model parallel and context
                parallel.
        """

        if not HAVE_FLA:
            raise ImportError(
                "FLA is not installed. Please install it with `pip install flash-linear-attention`."
            )

        super().__init__(config)

        # Attributes from arguments
        self.layer_number = layer_number
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
        self.A_init_range = A_init_range
        self.use_qk_l2norm = use_qk_l2norm
        assert pg_collection is not None, "pg_collection must be provided for GatedDeltaNet"
        self.pg_collection = pg_collection
        self.cp_size = self.pg_collection.cp.size()
        self.tp_size = self.pg_collection.tp.size()
        self.sp_size = self.tp_size if config.sequence_parallel else 1

        # Attributes from config
        self.config = config
        self.hidden_size = config.hidden_size
        self.act_fn = config.activation_func
        self.activation = self.act_fn.__name__
        self.conv_kernel_dim = config.linear_conv_kernel_dim
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads
        self.qk_dim_local_tp = self.qk_dim // self.tp_size
        self.v_dim_local_tp = self.v_dim // self.tp_size

        # Input projection (hidden_states -> q, k, v, gate, beta, alpha)
        # TODO: for now, output gate is forced for GDN.
        # We may remove this restriction in the future.
        self.in_proj_dim = self.qk_dim * 2 + self.v_dim * 2 + self.num_value_heads * 2
        if self.config.fp8:
            fp8_align_size = get_fp8_align_size(self.config.fp8_recipe)
            assert self.in_proj_dim % fp8_align_size == 0, (
                "For FP8, the innermost dimension of the GDN layer "
                "input projection output tensor must be a multiple of 16."
            )
        self.in_proj = build_module(
            submodules.in_proj,
            self.hidden_size,
            self.in_proj_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
        )

        # Conv1d for QKV
        self.conv_dim = self.qk_dim * 2 + self.v_dim
        self.conv_dim_local_tp = self.conv_dim // self.tp_size

        # weight shape: [conv_dim, 1, d_conv]
        # bias shape: [conv_dim]
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim_local_tp,
            out_channels=self.conv_dim_local_tp,
            bias=conv_bias,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim_local_tp,
            padding=self.conv_kernel_dim - 1,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        setattr(self.conv1d.weight, "tensor_model_parallel", True)
        if conv_bias:
            setattr(self.conv1d.bias, "tensor_model_parallel", True)

        # Time step projection (discretization)
        self.num_v_heads_local_tp = self.num_value_heads // self.tp_size
        # dt_bias parameter
        self.dt_bias = nn.Parameter(
            torch.empty(
                self.num_v_heads_local_tp,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.dt_bias, "tensor_model_parallel", True)
        # A_log parameter
        self.A_log = nn.Parameter(
            torch.empty(
                self.num_v_heads_local_tp,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.A_log, "tensor_model_parallel", True)
        
        if self.config.deterministic_mode:
            from mindspeed.core.ssm.chunk_gated_delta_rule import torch_chunk_gated_delta_rule
            self.gated_delta_rule = torch_chunk_gated_delta_rule
        else:
            self.gated_delta_rule = chunk_gated_delta_rule

        # Output layernorm before projection
        self.out_norm = build_module(
            submodules.out_norm,
            config=self.config,
            hidden_size=self.value_head_dim,
            eps=self.config.layernorm_epsilon,
        )

        self.out_proj = build_module(
            submodules.out_proj,
            self.v_dim,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=self.pg_collection.tp,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        if self.config.perform_initialization:
            with get_cuda_rng_tracker().fork():
                # conv1d.weight
                if self.conv_init is not None:
                    nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
                # dt_bias
                torch.ones(
                    self.num_v_heads_local_tp,
                    out=self.dt_bias.data,
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                )
                # A_log
                A = torch.empty(
                    self.num_v_heads_local_tp,
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                ).uniform_(*self.A_init_range)
                self.A_log.data.copy_(torch.log(A))

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        """
        Perform a forward pass through the GDN module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) GDN output and bias.

        """
        # TODO: Deal with attention_mask

        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size * self.cp_size

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "GDN does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError("GDN does not support inference for now.")

        if causal_conv1d is None and packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            raise RuntimeError(
                "THD (packed sequence) scenario only supports FLA fused operators, "
                "but FLA is not available. Please install the flash-linear-attention package."
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            assert batch == 1, "Packed sequence expects batch dimension to be 1"
            assert (
                not self.config.deterministic_mode
            ), "Packed sequence does not support deterministic mode."

            # Resolve cu_seqlens with alignment padding handling.
            cu_seqlens_q, cu_seqlens_q_list = self._resolve_cu_seqlens(
                packed_seq_params.cu_seqlens_q_padded,
                packed_seq_params.cu_seqlens_q,
                seq_len,
                "cu_seqlens_q",
                cp_size=self.cp_size,
            )
            cu_seqlens_kv, _ = self._resolve_cu_seqlens(
                packed_seq_params.cu_seqlens_kv_padded,
                packed_seq_params.cu_seqlens_kv,
                seq_len,
                "cu_seqlens_kv",
                cp_size=self.cp_size,
            )
            assert torch.equal(cu_seqlens_q, cu_seqlens_kv), (
                "Currently only support cu_seqlens_q equals to cu_seqlens_kv, "
                f"but got cu_seqlens_q={cu_seqlens_q} and cu_seqlens_kv={cu_seqlens_kv}"
            )
            num_packed_seqs = cu_seqlens_q.shape[0] - 1
            assert num_packed_seqs > 0, (
                "Number of packed sequences must be greater than 0, "
                f"but got cu_seqlens_q={cu_seqlens_q} and cu_seqlens_kv={cu_seqlens_kv}"
            )
        else:
            cu_seqlens_q = None
            cu_seqlens_q_list = None
            cu_seqlens_kv = None

        # Input projection
        nvtx_range_push(suffix="in_proj")
        qkvzba, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        # CP All to All: CP to HP
        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            unpacked_qkvzba = _unpack_sequence(qkvzba, cu_seqlens_q // self.cp_size, dim=0)
            outputs = []
            for qkvzba_i in unpacked_qkvzba:
                qkvzba_i = tensor_a2a_cp2hp(
                    qkvzba_i,
                    seq_dim=0,
                    head_dim=-1,
                    cp_group=self.pg_collection.cp,
                    split_sections=[
                        self.qk_dim_local_tp,
                        self.qk_dim_local_tp,
                        self.v_dim_local_tp,
                        self.v_dim_local_tp,
                        self.num_value_heads // self.tp_size,
                        self.num_value_heads // self.tp_size,
                    ],
                    undo_attention_load_balancing=False,
                )
                outputs.append(qkvzba_i)
            qkvzba = torch.cat(outputs, dim=0)
            # Per-segment zigzag undo (Megatron CP packs each segment with
            # zigzag load balancing across CP ranks). After cp->hp a2a, each
            # segment is laid out as [zigzag chunks] and must be permuted into
            # natural order before the recurrent gated_delta_rule runs over it.
            qkvzba = _undo_load_balancing_per_segment(
                qkvzba, cu_seqlens_q, cp_size=self.cp_size, seq_dim=0,
                cu_seqlens_list=cu_seqlens_q_list,
            )
        else:
            qkvzba = tensor_a2a_cp2hp(
                qkvzba,
                seq_dim=0,
                head_dim=-1,
                cp_group=self.pg_collection.cp,
                split_sections=[
                    self.qk_dim_local_tp,
                    self.qk_dim_local_tp,
                    self.v_dim_local_tp,
                    self.v_dim_local_tp,
                    self.num_value_heads // self.tp_size,
                    self.num_value_heads // self.tp_size,
                ],
            )

        # Transpose: s b x --> b s x
        # From sbhd to bshd format
        qkvzba = qkvzba.transpose(0, 1)

        # Split, reorder, and reshape the tensor into q, k, v, gate, beta, alpha
        qkv, gate, beta, alpha = torch.split(
            qkvzba,
            [
                (self.qk_dim_local_tp * 2 + self.v_dim_local_tp) // self.cp_size,
                self.v_dim_local_tp // self.cp_size,
                self.num_value_heads // self.tp_size // self.cp_size,
                self.num_value_heads // self.tp_size // self.cp_size,
            ],
            dim=-1,
        )
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
        beta = beta.reshape(batch, seq_len, -1)
        alpha = alpha.reshape(batch, seq_len, -1)

        # Convolution on qkv
        nvtx_range_push(suffix="conv1d")
        seq_len = qkv.shape[1]
        qkv_channels_split_sections = [
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
        ]
        conv1d_weight = get_parameter_local_cp(
            self.conv1d.weight,
            dim=0,
            cp_group=self.pg_collection.cp,
            split_sections=qkv_channels_split_sections,
        )
        conv1d_bias = (
            get_parameter_local_cp(
                self.conv1d.bias,
                dim=0,
                cp_group=self.pg_collection.cp,
                split_sections=qkv_channels_split_sections,
            )
            if self.conv_bias
            else None
        )
        if causal_conv1d is None or self.config.deterministic_mode:
            qkv = qkv.transpose(1, 2).contiguous()  # b, s, d -> b, d, s
            conv_out = F.conv1d(
                input=qkv,  # Torch-native only accept [b, d, s] format input
                weight=conv1d_weight,
                bias=conv1d_bias,
                stride=self.conv1d.stride,
                padding=self.conv1d.padding,
                dilation=self.conv1d.dilation,
                groups=self.conv_dim_local_tp // self.cp_size,
            )
            qkv = self.act_fn(conv_out[..., :seq_len])
            qkv = qkv.transpose(1, 2)  # b, d, s -> b, s, d
        else:
            assert self.activation in [None, "silu", "swish"]
            qkv, _ = causal_conv1d(
                x=qkv,  # FLA conv1d accepts [b, s, d] format input
                weight=conv1d_weight.squeeze(1),  # d, 1, w -> d, w
                bias=conv1d_bias,
                activation=self.activation,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=cu_seqlens_q,
            )
        nvtx_range_pop(suffix="conv1d")

        # Prepare QKV tensors (split, reshape, L2 norm, repeat_interleave, contiguous)
        nvtx_range_push(suffix="prepare_qkv_for_gated_delta_rule")
        query, key, value, gate, beta, alpha = self._prepare_qkv_for_gated_delta_rule(
            qkv, gate, beta, alpha, batch, seq_len
        )
        nvtx_range_pop(suffix="prepare_qkv_for_gated_delta_rule")

        # Calculate g and beta
        nvtx_range_push(suffix="g_and_beta")
        A_log_local_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=self.pg_collection.cp)
        dt_bias_local_cp = get_parameter_local_cp(
            self.dt_bias, dim=0, cp_group=self.pg_collection.cp
        )
        g, beta = self._compute_g_and_beta(A_log_local_cp, dt_bias_local_cp, alpha, beta)
        nvtx_range_pop(suffix="g_and_beta")

        nvtx_range_push(suffix="gated_delta_rule")
        core_attn_out, last_recurrent_state = self.gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=cu_seqlens_q,
        )
        nvtx_range_pop(suffix="gated_delta_rule")

        # RMSNorm
        nvtx_range_push(suffix="gated_norm")
        if core_attn_out.shape[1] < gate.shape[1]:
            core_attn_out = F.pad(core_attn_out, (0, 0, 0, 0, 0, gate.shape[1] - core_attn_out.shape[1]))
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix="gated_norm")

        # Transpose: b s x --> s b x
        # From bshd back to sbhd format
        norm_out = norm_out.reshape(batch, seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()

        # CP all to all: HP to CP
        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # Per-segment zigzag redo: undo above brought each segment into
            # natural order, so before the hp->cp a2a we have to restore the
            # zigzag layout that the downstream code expects.
            norm_out = _redo_load_balancing_per_segment(
                norm_out, cu_seqlens_q, cp_size=self.cp_size, seq_dim=0,
                cu_seqlens_list=cu_seqlens_q_list,
            )
            unpacked_norm_out = _unpack_sequence(norm_out, cu_seqlens_q, dim=0)
            outputs = []
            for norm_out_i in unpacked_norm_out:
                norm_out_i = tensor_a2a_hp2cp(
                    norm_out_i, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp,
                    redo_attention_load_balancing=False,
                )
                outputs.append(norm_out_i)
            norm_out = torch.cat(outputs, dim=0)
        else:
            norm_out = tensor_a2a_hp2cp(
                norm_out, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp
            )

        # Output projection
        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")

        return out, out_bias

    @jit_fuser
    def _apply_gated_norm(self, x, gate):
        # Output Norm
        x_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1])
        y = self.out_norm(x)
        # Output gate
        gate = gate.reshape(-1, gate.shape[-1])
        y = y * self.act_fn(gate.float())
        y = y.to(x_dtype)
        return y

    @jit_fuser
    def _prepare_qkv_for_gated_delta_rule(self, qkv, gate, beta, alpha, batch, seq_len):
        """
        Prepare query, key, value, gate, beta, alpha tensors for gated delta rule.
        Fuses split, reshape, L2 norm, repeat_interleave, and contiguous operations.
        """
        # Split qkv into query_key and value
        query_key, value = torch.split(
            qkv,
            [2 * self.qk_dim_local_tp // self.cp_size, self.v_dim_local_tp // self.cp_size],
            dim=-1,
        )

        # Reshape query_key and value
        query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)

        # Apply L2 norm to query and key
        if self.use_qk_l2norm:
            args = get_args()
            if args.use_naive_l2norm:
                query_key = naive_l2norm(query_key.contiguous())
            else:
                query_key = l2norm(query_key.contiguous())

        # Split query and key
        split_size = self.qk_dim_local_tp // self.key_head_dim // self.cp_size
        query, key = torch.split(query_key, [split_size, split_size], dim=2)

        # Expand query and key if needed (grouped query attention)
        if self.num_value_heads // self.num_key_heads > 1:
            repeat_factor = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        # Make all tensors contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous()
        alpha = alpha.contiguous()

        return query, key, value, gate, beta, alpha

    @jit_fuser
    def _compute_g_and_beta(self, A_log_local_cp, dt_bias_local_cp, alpha, beta):
        """
        Compute g (decay) and beta (sigmoid) for gated delta rule.
        Fuses exp, softplus, mul, neg, and sigmoid operations.
        """
        g = -A_log_local_cp.exp() * F.softplus(alpha.float() + dt_bias_local_cp)  # In fp32
        beta = beta.sigmoid()
        return g, beta

    def _resolve_cu_seqlens(
        self, cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name, cp_size: int = 1
    ):
        """Resolve cu_seqlens for packed sequence all-to-all.

        Preferred contract (produced by
        ``mindspeed.core.ssm.packed_seq_utils.preprocess_packed_seqs`` /
        ``build_packed_seq_params``):

        * ``cu_seqlens_padded`` is preferred; ``cu_seqlens_actual`` is the
          fallback when the caller did not pre-compute the aligned version.
        * The chosen tensor starts with ``0`` and ends with ``total_seq_len``.
        * Every per-segment length is divisible by ``cp_size``.

        Compatibility fall-backs (kept for upstream callers that still feed
        the raw ``actual_seq_len`` tensor without a leading zero, e.g. the
        ``gpt_forward_wrapper`` in ``mindspeed.core.models.gpt.gpt_model`` and
        ``mindspeed.core.transformer.flash_attention.reset_attention_mask``):

        * If the first element is non-zero, we either prepend ``0`` (when the
          last element already equals ``total_seq_len``) or subtract the
          first-element offset (when ``last - first == total_seq_len``). This
          mirrors the historical behaviour and avoids breaking existing
          training scripts that still rely on the unaligned input layout.

        The divisibility check is still mandatory because GDN's downstream
        all-to-all + chunked kernels cannot recover from a non-aligned split.
        """
        cu_seqlens = cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens_actual
        cu_seqlens = cu_seqlens.reshape(-1)

        # Compatibility path: normalise a non-zero leading element so legacy
        # data pipelines that store cumulative seqlens *without* a leading 0
        # keep working. We need an .item() here to decide the branch, but it
        # is the only D2H on the fast path and is unavoidable when the input
        # layout itself is ambiguous.
        if cu_seqlens.numel() > 0 and int(cu_seqlens[0].item()) != 0:
            total_cu = int(cu_seqlens[-1].item())
            if total_cu == total_seq_len:
                cu_seqlens = torch.cat(
                    [
                        torch.zeros(
                            1, dtype=cu_seqlens.dtype, device=cu_seqlens.device
                        ),
                        cu_seqlens,
                    ]
                )
            elif total_cu - int(cu_seqlens[0].item()) == total_seq_len:
                cu_seqlens = cu_seqlens - cu_seqlens[0]

        # Validation runs on device; only on failure do we materialise scalars
        # to build a descriptive error message.
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        last_matches_total = cu_seqlens[-1].eq(total_seq_len)
        all_divisible = (seq_lengths % cp_size).eq(0).all()
        ok = last_matches_total & all_divisible

        if not bool(ok):
            last = int(cu_seqlens[-1].item())
            if last != total_seq_len:
                raise ValueError(
                    f"GDN: {name}[-1]={last} does not match "
                    f"total_sequence_length={total_seq_len}. "
                    f"({cu_seqlens_padded=}, {cu_seqlens_actual=})."
                )
            raise ValueError(
                f"All per-sequence lengths in {name} must be divisible by "
                f"cp_size={cp_size}, but got lengths: {seq_lengths.tolist()}. "
            )

        # Single D2H: downstream functions receive the list form to avoid
        # redundant .cpu().tolist() in causal_conv1d / flash_gated_delta_rule.
        cu_seqlens_list = cu_seqlens.tolist()
        return cu_seqlens, cu_seqlens_list

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        """Provide a sharded state dictionary for distributed checkpointing."""
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = {}
        # Parameters
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map={
                "A_log": 0,
                "dt_bias": 0,
            },  # parameters sharded across TP
            sharded_offsets=sharded_offsets,
            tp_group=(tp_group if tp_group is not None else self.pg_collection.tp),
            dp_cp_group=metadata['dp_cp_group'],
        )
        # Submodules
        tp_group = tp_group if tp_group is not None else self.pg_collection.tp
        for name, module in self.named_children():
            if name == "conv1d":
                # Add TP sharding for Conv1d
                module_sd = module.state_dict(prefix="", keep_vars=True)
                tp_sharding_map = {f"weight": 0}
                if self.conv_bias:
                    tp_sharding_map[f"bias"] = 0
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd,
                    f"{prefix}{name}.",
                    tp_sharding_map,
                    sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=metadata['dp_cp_group'],
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
                )

            sharded_state_dict.update(module_sharded_sd)

        # At this point the TP sharding is correctly defined for each tensor, but some of the
        # tensors must be additionally split into separate parts
        in_proj_dim_local_tp = self.in_proj_dim // self.tp_size
        assert sharded_state_dict[f"{prefix}in_proj.weight"].data.size(0) == in_proj_dim_local_tp, (
            in_proj_dim_local_tp,
            sharded_state_dict[f"{prefix}in_proj.weight"],
        )

        sharded_state_dict[f"{prefix}in_proj.weight"] = _split_tensor_factory(
            sharded_state_dict[f"{prefix}in_proj.weight"],
            [
                self.qk_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
                self.v_dim_local_tp,
                self.num_value_heads // self.tp_size,
                self.num_value_heads // self.tp_size,
            ],
            ["query", "key", "value", "z", "beta", "alpha"],
            0,
        )

        conv_layer_name_list = ["conv1d.weight"]
        assert (
            sharded_state_dict[f"{prefix}conv1d.weight"].data.size(0) == self.conv_dim_local_tp
        ), (self.conv_dim_local_tp, sharded_state_dict[f"{prefix}conv1d.weight"])
        if self.conv_bias:
            conv_layer_name_list.append("conv1d.bias")
            assert (
                sharded_state_dict[f"{prefix}conv1d.bias"].data.size(0) == self.conv_dim_local_tp
            ), (self.conv_dim_local_tp, sharded_state_dict[f"{prefix}conv1d.bias"])
        for conv_layer_name in conv_layer_name_list:
            sharded_state_dict[f"{prefix}{conv_layer_name}"] = _split_tensor_factory(
                sharded_state_dict[f"{prefix}{conv_layer_name}"],
                [self.qk_dim_local_tp, self.qk_dim_local_tp, self.v_dim_local_tp],
                ["query", "key", "value"],
                0,
            )

        return sharded_state_dict

    def backward_dw(self):
        """Execute weight gradient computation for all linear layers."""
        self._backward_in_proj()
        self._backward_out_proj()

    def _backward_in_proj(self):
        """Computes weight gradients of input projection layer."""
        self.in_proj.backward_dw()

    def _backward_out_proj(self):
        """Computes weight gradients of output projection layer."""
        self.out_proj.backward_dw()


def _unpack_sequence(x, cu_seqlens, dim=1):
    unpacked_x = []
    num_seqs = cu_seqlens.shape[0] - 1
    for i in range(num_seqs):
        idx_start = 0 if i == 0 else cu_seqlens[i].item()
        idx_end = cu_seqlens[i + 1].item()
        chunked_index = [slice(None)] * dim + [slice(idx_start, idx_end)]
        unpacked_x.append(x[tuple(chunked_index)])
    return unpacked_x


def _undo_load_balancing_per_segment(
    tensor: torch.Tensor, cu_seqlens: torch.Tensor, cp_size: int, seq_dim: int = 0,
    cu_seqlens_list: Optional[List[int]] = None
) -> torch.Tensor:
    """Convert zigzag layout to natural order, segment by segment.

    The per-segment zigzag layout that Megatron CP produces concatenates
    ``2*cp_size`` chunks per segment with order
    ``[c0, c_{2c-1}, c1, c_{2c-2}, ..., c_{c-1}, c_c]`` (cp_size=2 gives
    ``[c0, c3, c1, c2]``).  This function reorders each segment back to
    ``[c0, c1, c2, c3]`` so that the gated_delta_rule recurrent kernel sees
    the natural causal order.

    ``cu_seqlens`` here must be the *global* (HP) cumulative segment
    boundaries -- the same value that GDN feeds to ``gated_delta_rule``.
    Each segment length is required to be a multiple of ``2 * cp_size``.

    When ``cu_seqlens_list`` is provided, it is used directly instead of
    per-element ``.item()`` calls on ``cu_seqlens``, avoiding D2H transfers.
    """
    if cp_size == 1:
        return tensor

    if seq_dim != 0:
        raise NotImplementedError(
            f"_undo_load_balancing_per_segment only supports seq_dim==0, got {seq_dim}"
        )

    num_chunks_div_2 = cp_size
    num_chunks = num_chunks_div_2 * 2

    num_seqs = cu_seqlens.shape[0] - 1
    if num_seqs <= 0:
        return tensor

    reordered = []
    for i in range(num_seqs):
        if cu_seqlens_list is not None:
            start = cu_seqlens_list[i]
            end = cu_seqlens_list[i + 1]
        else:
            start = 0 if i == 0 else int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
        seg = tensor[start:end]
        if seg.size(0) == 0:
            reordered.append(seg)
            continue
        if seg.size(0) % num_chunks != 0:
            raise ValueError(
                f"GDN: segment {i} length {seg.size(0)} is not divisible by "
                f"2 * cp_size = {num_chunks}; rebuild cu_seqlens with aligned "
                f"per-sequence lengths before enabling packed sequence GDN."
            )
        chunks = torch.chunk(seg, chunks=num_chunks, dim=0)
        order = [2 * j for j in range(num_chunks_div_2)] + [
            num_chunks - 2 * j - 1 for j in range(num_chunks_div_2)
        ]
        reordered.append(torch.cat([chunks[j] for j in order], dim=0))

    return torch.cat(reordered, dim=0)


def _redo_load_balancing_per_segment(
    tensor: torch.Tensor, cu_seqlens: torch.Tensor, cp_size: int, seq_dim: int = 0,
    cu_seqlens_list: Optional[List[int]] = None
) -> torch.Tensor:
    """Inverse of :func:`_undo_load_balancing_per_segment`.

    Converts a per-segment natural order ``[c0, c1, c2, c3]`` back into the
    zigzag layout ``[c0, c3, c1, c2]`` (cp_size=2).

    When ``cu_seqlens_list`` is provided, it is used directly instead of
    per-element ``.item()`` calls on ``cu_seqlens``, avoiding D2H transfers.
    """
    if cp_size == 1:
        return tensor

    if seq_dim != 0:
        raise NotImplementedError(
            f"_redo_load_balancing_per_segment only supports seq_dim==0, got {seq_dim}"
        )

    num_chunks_div_2 = cp_size
    num_chunks = num_chunks_div_2 * 2

    num_seqs = cu_seqlens.shape[0] - 1
    if num_seqs <= 0:
        return tensor

    reordered = []
    for i in range(num_seqs):
        if cu_seqlens_list is not None:
            start = cu_seqlens_list[i]
            end = cu_seqlens_list[i + 1]
        else:
            start = 0 if i == 0 else int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
        seg = tensor[start:end]
        if seg.size(0) == 0:
            reordered.append(seg)
            continue
        if seg.size(0) % num_chunks != 0:
            raise ValueError(
                f"GDN: segment {i} length {seg.size(0)} is not divisible by "
                f"2 * cp_size = {num_chunks}; rebuild cu_seqlens with aligned "
                f"per-sequence lengths before enabling packed sequence GDN."
            )
        chunks = torch.chunk(seg, chunks=num_chunks, dim=0)
        order = [None] * num_chunks
        order[::2] = range(num_chunks_div_2)
        order[1::2] = reversed(range(num_chunks_div_2, num_chunks))
        reordered.append(torch.cat([chunks[j] for j in order], dim=0))

    return torch.cat(reordered, dim=0)


####################
# Context parallel utilities
####################
def get_parameter_local_cp(
    param: torch.Tensor,
    dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
) -> torch.Tensor:
    """Get the local parameter for the current context parallel rank.

    Args:
        param (torch.Tensor): The entire parameter to get the local parameter for.
        dim (int): The dimension to split the parameter along. Usually the dimension of head.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
        split_sections (Optional[List[int]]): If not None,
            first split the parameter along the dimension dim into sections,
            then get the local hidden parallel weights separately,
            finally concatenate the local hidden parallel weights along the dimension dim.

    Returns:
        torch.Tensor: The local parameter for the current context parallel rank.
    """

    cp_size = cp_group.size()
    cp_rank = cp_group.rank()

    # No need to split if CP size is 1.
    if cp_size == 1:
        return param

    # Split first if needed.
    if split_sections is not None:
        inputs = torch.split(param, split_sections, dim=dim)
        outputs = []
        for p in inputs:
            p = get_parameter_local_cp(p, dim, cp_group)
            outputs.append(p)
        return torch.cat(outputs, dim=dim)

    # Slice the parameter.
    slices = [slice(None)] * param.dim()
    dim_size = param.size(dim=dim)
    slices[dim] = slice(cp_rank * dim_size // cp_size, (cp_rank + 1) * dim_size // cp_size)
    param = param[tuple(slices)]
    return param


def tensor_a2a_cp2hp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
    undo_attention_load_balancing: bool = True,
):
    """All-to-all context parallel to hidden parallel.

    Args:
        tensor (torch.Tensor): The tensor to all-to-all.
            Currently only support (seq_len, batch, head_dim) shaped tensor.
        seq_dim (int): The dimension of sequence length. Currently only supports seq_dim == 0.
        head_dim (int): The dimension of head. Currently only supports head_dim == -1 or 2.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
        split_sections (Optional[List[int]]): If not None, split the tensor along the dimension
            head_dim into sections first, then do all-to-all for each section separately,
            finally concatenate the separated tensors along the dimension head_dim.
        undo_attention_load_balancing (bool): Whether to undo the attention load balancing of CP.

    Returns:
        torch.Tensor: The all-to-all tensor.
    """

    cp_size = cp_group.size()

    # No need to all-to-all if CP size is 1.
    if cp_size == 1:
        return tensor

    # Limitations of mamba_context_parallel._all_to_all_cp2hp.
    assert seq_dim == 0, (
        f"tensor_a2a_cp2hp only supports seq_dim == 0 for now, but got seq_dim={seq_dim}"
    )
    assert (
        head_dim == -1 or head_dim == 2
    ), f"tensor_a2a_cp2hp only supports head_dim == -1 or 2 for now, but got head_dim={head_dim}"
    assert (
        tensor.dim() == 3
    ), f"tensor_a2a_cp2hp only supports 3-d input tensor for now, but got tensor.dim()={tensor.dim()}"

    # Split first if needed.
    if split_sections is not None:
        inputs = torch.split(tensor, split_sections, dim=head_dim)
        outputs = []
        for x in inputs:
            x = tensor_a2a_cp2hp(
                x,
                seq_dim=seq_dim,
                head_dim=head_dim,
                cp_group=cp_group,
                undo_attention_load_balancing=False,
            )
            outputs.append(x)
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_cp2hp(tensor, cp_group)

    # Undo attention load balancing last if needed.
    if undo_attention_load_balancing:
        tensor = _undo_attention_load_balancing(tensor, cp_size)
    return tensor


def tensor_a2a_hp2cp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
    redo_attention_load_balancing: bool = True,
):
    """All-to-all hidden parallel to context parallel.

    Args:
        tensor (torch.Tensor): The tensor to all-to-all.
            Currently only support (seq_len, batch, head_dim) shaped tensor.
        seq_dim (int): The dimension of sequence length. Currently only supports seq_dim == 0.
        head_dim (int): The dimension of head. Currently only supports head_dim == -1 or 2.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
        split_sections (Optional[List[int]]): If not None, first split the tensor along the
            dimension head_dim into sections, then do all-to-all for each section separately,
            finally concatenate the separated tensors along the dimension head_dim.
        redo_attention_load_balancing (bool): Whether to redo the attention load balancing of HP.

    Returns:
        torch.Tensor: The all-to-all tensor.
    """

    cp_size = cp_group.size()

    # No need to all-to-all if CP size is 1.
    if cp_size == 1:
        return tensor

    # Limitations of mamba_context_parallel._all_to_all_hp2cp.
    assert seq_dim == 0, (
        f"tensor_a2a_hp2cp only supports seq_dim == 0 for now, but got seq_dim={seq_dim}"
    )
    assert (
        head_dim == -1 or head_dim == 2
    ), (
        f"tensor_a2a_hp2cp only supports head_dim == -1 or 2 for now, but got head_dim={head_dim}"
    )
    assert (
        tensor.dim() == 3
    ), (
        f"tensor_a2a_hp2cp only supports 3-d input tensor for now, but got tensor.dim()={tensor.dim()}"
    )

    # Redo attention load balancing first if needed.
    if redo_attention_load_balancing:
        tensor = _redo_attention_load_balancing(tensor, cp_size)

    # Split first if needed.
    if split_sections is not None:
        inputs = torch.split(tensor, split_sections, dim=head_dim)
        outputs = []
        for x in inputs:
            x = tensor_a2a_hp2cp(
                x,
                seq_dim=seq_dim,
                head_dim=head_dim,
                cp_group=cp_group,
                redo_attention_load_balancing=False,
            )
            outputs.append(x)
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_hp2cp(tensor, cp_group)

    return tensor
