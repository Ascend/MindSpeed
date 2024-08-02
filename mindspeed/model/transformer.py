# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team. All rights reserved.
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
import math
import threading
from functools import wraps

import torch
import torch_npu
import torch.nn.functional as F
from megatron.training import get_args, get_tokenizer
from megatron.core import parallel_state, mpu, tensor_parallel
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.legacy.model.utils import openai_gelu, erf_gelu, get_norm
from megatron.legacy.model.transformer import ParallelMLP, ParallelTransformer, ParallelTransformerLayer
from megatron.core.enums import ModelType
from megatron.legacy.model.enums import AttnType, AttnMaskType, LayerType
from megatron.legacy.model.transformer import _get_num_layers, _get_layer_type
from megatron.legacy.model.fused_bias_gelu import bias_gelu_impl

from mindspeed.core.context_parallel.ulysses_context_parallel import UlyssesContextAttention
from mindspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel
from mindspeed.core.parallel_state import (get_context_parallel_group_for_hybrid_ulysses,
                                             get_context_parallel_group_for_hybrid_ring,
                                             get_context_parallel_for_hybrid_ring_world_size,
                                             get_context_parallel_for_hybrid_ring_rank,
                                             get_context_parallel_for_hybrid_ring_global_ranks)
from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
from mindspeed.core.tensor_parallel.layers import Nd_ParallelLinear
from mindspeed.core.tensor_parallel.checkpoint_manager import get_pipeline_checkpoint_manager
from mindspeed.model.alibi_mask import AlibiForFusionAttnSingleton, get_slopes

try:
    from einops import rearrange
except ImportError:
    rearrange = None

_GLOBAL_ATTN_MASK = None


class Alibi:
    _instance = None
    alibi = None
    matmul_result = None
    output_size = None
    lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance:
            return cls._instance
        else:
            with cls.lock:
                cls._instance = super().__new__(cls)
                return cls._instance


def _get_inverted_mask(attention_mask, alibi):
    inverted_mask = attention_mask.to(alibi.dtype)
    inverted_mask = inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), float("-inf")
    )
    return inverted_mask.to(alibi.device) + alibi.unsqueeze(0)


def _build_alibi_tensor(max_seq_len, num_attention_heads, square_alibi_mask, fill_neg_inf):
    def _fill_with_neg_inf(t):
        """FP16-compatible function that fills a tensor with -inf."""
        return t.float().fill_(float("-inf")).type_as(t)

    def _buffered_future_mask(maxpos, alibi, attn_heads):
        _future_mask = torch.triu(_fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1)
        _future_mask = _future_mask.unsqueeze(0) + alibi
        return _future_mask[:attn_heads, :maxpos, :maxpos]

    slopes = torch.Tensor(get_slopes(num_attention_heads))
    if square_alibi_mask:
        position_point = torch.arange(max_seq_len) - max_seq_len + 1
        position_point = position_point.unsqueeze(0).unsqueeze(0).expand(num_attention_heads, max_seq_len, -1)
        diag = torch.diag(position_point[0])
        position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
        alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
    else:
        alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(
            num_attention_heads, -1, -1)

    # Select the part of the tensor that corresponds to our tensor parallel index.
    tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_index = parallel_state.get_tensor_model_parallel_rank()
    alibi = alibi.reshape((tp_world_size, -1, *alibi.shape[1:]))[tp_index]

    if fill_neg_inf:
        return _buffered_future_mask(max_seq_len, alibi, num_attention_heads)

    return alibi


def core_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *arg, **kwargs):
        fn(self, *arg, **kwargs)

        args = get_args()
        self.hidden_size_per_partition = self.hidden_size_per_partition // args.context_parallel_size
        self.square_alibi_mask = args.square_alibi_mask
        self.fill_neg_inf = args.fill_neg_inf
        self.beta = 1.0
        if self.apply_query_key_layer_scaling:
            self.beta = 1.0 / self.layer_number
        if args.position_embedding_type == 'alibi':
            self.alibi = Alibi()
            alibi = _build_alibi_tensor(args.seq_length,
                                        args.num_attention_heads,
                                        args.square_alibi_mask,
                                        args.fill_neg_inf
                                        ).to(torch.cuda.current_device())
            if args.params_dtype == torch.float16:
                alibi = alibi.to(torch.float16)
            elif args.params_dtype == torch.bfloat16:
                alibi = alibi.to(torch.bfloat16)
            self.alibi.alibi = alibi
        else:
            self.alibi = None

    return wrapper


def core_attention_forward(self, query_layer, key_layer, value_layer, attention_mask):
    # ===================================
    # Raw attention scores. [b, np, s, s]
    # ===================================

    # [b, np, sq, sk]
    output_size = (query_layer.size(1),
                   query_layer.size(2),
                   query_layer.size(0),
                   key_layer.size(0))

    # [sq, b, np, hn] -> [sq, b * np, hn]
    query_layer = query_layer.reshape(output_size[2],
                                      output_size[0] * output_size[1], -1)
    # [sk, b, np, hn] -> [sk, b * np, hn]
    key_layer = key_layer.view(output_size[3],
                               output_size[0] * output_size[1], -1)

    if self.alibi is None:
        matmul_input_buffer = mpu.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]),
            query_layer.dtype, "mpu")

        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),
            key_layer.transpose(0, 1).transpose(1, 2),
            beta=0.0, alpha=(1.0 / self.norm_factor))
    else:
        if self.alibi.matmul_result is None or self.alibi.output_size != output_size:
            args = get_args()

            self.alibi.output_size = output_size
            alibi = _build_alibi_tensor(args.seq_length,
                                        args.num_attention_heads,
                                        args.square_alibi_mask,
                                        args.fill_neg_inf
                                        ).to(torch.cuda.current_device())
            if args.params_dtype == torch.float16:
                alibi = alibi.to(torch.float16)
            elif args.params_dtype == torch.bfloat16:
                alibi = alibi.to(torch.bfloat16)
            self.alibi.alibi = alibi

            if self.fill_neg_inf:
                _alibi = self.alibi.alibi[:, :output_size[3], :output_size[3]]
                attention_mask = attention_mask.repeat(output_size[0], 1, 1, 1)[:output_size[0], :, :, :]
                self.alibi.matmul_result = _get_inverted_mask(attention_mask, _alibi).view(-1, output_size[2],
                                                                                           output_size[2]).contiguous()
            else:
                self.alibi.matmul_result = self.alibi.alibi[:, :, :output_size[3]].repeat(output_size[0], 1, 1)

        q_trans = query_layer.transpose(0, 1).contiguous()
        k_trans = key_layer.transpose(0, 1).transpose(1, 2).contiguous()
        matmul_result = self.beta * self.alibi.matmul_result + torch.bmm(q_trans, k_trans) * (1.0 / self.norm_factor)

        # change view to [b, np, sq, sk]
    attention_scores = matmul_result.view(*output_size)

    # ===========================
    # Attention probs and dropout
    # ===========================

    # attention scores and attention mask [b, np, sq, sk]
    if self.square_alibi_mask:
        attention_scores = torch.max(
            attention_scores, torch.tensor(torch.finfo(attention_scores.dtype).min)
        )
        attention_probs = torch.nn.functional.softmax(attention_scores, -1)
    else:
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    if not self.sequence_parallel:
        with tensor_parallel.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)
    else:
        attention_probs = self.attention_dropout(attention_probs)

    # =========================
    # Context layer. [sq, b, hp]
    # =========================

    # value_layer -> context layer.
    # [sk, b, np, hn] --> [b, np, sq, hn]

    # context layer shape: [b, np, sq, hn]
    output_size = (value_layer.size(1),
                   value_layer.size(2),
                   query_layer.size(0),
                   value_layer.size(3))

    # change view [sk, b * np, hn]
    value_layer = value_layer.view(value_layer.size(0),
                                   output_size[0] * output_size[1], -1)

    # change view [b * np, sq, sk]
    attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                           output_size[2], -1)

    # matmul: [b * np, sq, hn]
    context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

    # change view [b, np, sq, hn]
    context_layer = context_layer.view(*output_size)

    # [b, np, sq, hn] --> [sq, b, np, hn]
    context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

    # [sq, b, np, hn] --> [sq, b, hp]
    new_context_layer_shape = context_layer.size()[:-2] + \
                              (self.hidden_size_per_partition,)
    context_layer = context_layer.view(*new_context_layer_shape)

    return context_layer


def parallel_transformer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        def build_layer(model_type, config, layer_number, layer_type=LayerType.encoder, self_attn_mask_type=AttnMaskType.padding):
            current_layer_type = _get_layer_type(
                model_type, layer_type, self.retro_layer_numbers,
                layer_number)
            return ParallelTransformerLayer(
                config,
                layer_number,
                layer_type=current_layer_type,
                self_attn_mask_type=self_attn_mask_type,
                drop_path_rate=self.drop_path_rates[layer_number - 1])
        fn(self, *args, **kwargs)

        argument = get_args()
        if argument.automated_pipeline and argument.num_layer_list and argument.virtual_pipeline_model_parallel_size is None:
            start_layer_num = 1
            self.layers = torch.nn.ModuleList()
            for idx, value in enumerate(argument.num_layer_list):
                if parallel_state.get_pipeline_model_parallel_rank() == idx:
                    self.num_layers = value
                    for layer_num in range(start_layer_num, start_layer_num + value):
                        self.layers.append(build_layer(kwargs['model_type'], args[0], layer_num, self_attn_mask_type=kwargs['self_attn_mask_type']))
                start_layer_num += value
            self.layers = torch.nn.ModuleList(self.layers)

            # Update dropout rate for Retro encoder.
            if kwargs['model_type'] == ModelType.retro_encoder:
                for layer in self.layers:
                    if layer.self_attention.use_flash_attn:
                        layer.self_attention.core_attention_flash.dropout_p = \
                            torch.nn.Dropout(argument.retro_encoder_attention_dropout)
                    else:
                        layer.self_attention.core_attention.attention_dropout.p = \
                            argument.retro_encoder_attention_dropout
                    layer.hidden_dropout = argument.retro_encoder_hidden_dropout
    return wrapper


def set_attention_mask(attn_mask):
    global _GLOBAL_ATTN_MASK
    _GLOBAL_ATTN_MASK = attn_mask


def generate_attention_mask():
    global _GLOBAL_ATTN_MASK
    args = get_args()
    if args.use_flash_attn and (args.seq_length > 2048 or args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']):
        args.sparse_mode = 2
        _GLOBAL_ATTN_MASK = torch.triu(torch.ones([2048, 2048], dtype=bool, device=torch.cuda.current_device()), diagonal=1)
    else:
        args.sparse_mode = 0
        _GLOBAL_ATTN_MASK = (torch.tril(torch.ones([args.micro_batch_size, 1, args.seq_length, args.seq_length], dtype=bool, device=torch.cuda.current_device()), diagonal=-(args.pre_tockens + 1)) \
            + torch.triu(torch.ones([args.micro_batch_size, 1, args.seq_length, args.seq_length], dtype=bool, device=torch.cuda.current_device()), diagonal=args.next_tockens + 1))


def get_attention_mask():
    global _GLOBAL_ATTN_MASK
    args = get_args()
    if args.cp_attention_mask_type == 'causal' and _GLOBAL_ATTN_MASK is None:
        generate_attention_mask()
    return _GLOBAL_ATTN_MASK


def parallel_transformer_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, hidden_states, attention_mask, **kwargs):
        args = get_args()
        attention_mask = get_attention_mask()
        return fn(self, hidden_states, attention_mask, **kwargs)
    return wrapper


def parallel_mlp_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        self.layer_number = None
        _args = get_args()
        if _args.swiglu and _args.use_fused_swiglu:
            self.activation_func = fused_swiglu

        config = args[0]
        is_expert = kwargs.get('is_expert') if 'is_expert' in kwargs.keys() else False

        ffn_hidden_size = config.ffn_hidden_size
        if config.gated_linear_unit:
            ffn_hidden_size *= 2
        if _args.use_nd_matmul:
            self.dense_h_to_4h = Nd_ParallelLinear(
                config.hidden_size,
                ffn_hidden_size,
                config=config,
                init_method=config.init_method,
                bias=self.add_bias,
                skip_bias_add=True,
                input_is_parallel=True,
                is_expert=is_expert,
                matmul_id=1
            )
            self.dense_4h_to_h = Nd_ParallelLinear(
                config.ffn_hidden_size,
                config.hidden_size,
                config=config,
                init_method=config.output_layer_init_method,
                bias=self.add_bias,
                skip_bias_add=True,
                input_is_parallel=True,
                is_expert=is_expert,
                matmul_id=2
            )
        else:
            self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                ffn_hidden_size,
                config=config,
                init_method=config.init_method,
                bias=self.add_bias,
                gather_output=False,
                skip_bias_add=True,
                is_expert=is_expert
            )
            self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
                config.ffn_hidden_size,
                config.hidden_size,
                config=config,
                init_method=config.output_layer_init_method,
                bias=self.add_bias,
                skip_bias_add=True,
                input_is_parallel=True,
                is_expert=is_expert
            )
        if _args.use_nanopipe and parallel_state.get_pipeline_model_parallel_world_size() > 1 \
                    and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            setattr(self.dense_h_to_4h, "in_nano", True)
            setattr(self.dense_4h_to_h, "in_nano", True)
        # use dynamic property assignment to ADD pipe_experts attribution
        if not _args.swiglu:
            self.dense_h_to_4h.pipe_experts = _args.use_pipe_experts
            self.dense_4h_to_h.pipe_experts = _args.use_pipe_experts
    return wrapper


def should_recompute_activation(self):
    args = get_args()
    if not args.recompute_activation_function or self.layer_number is None:
        return False

    activation_recompute_layers = args.recompute_activation_function_num_layers
    vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    vpp_size = args.virtual_pipeline_model_parallel_size
    pp_size = args.transformer_pipeline_model_parallel_size

    if args.recompute_in_bubble or args.recompute_in_advance:
        pipeline_checkpoint_manager = get_pipeline_checkpoint_manager(vpp_size)
        if pipeline_checkpoint_manager.chunk_do_recompute:
            return False
        elif args.recompute_in_bubble:
            return True

    if vpp_size is not None:
        layer_per_chunk = args.num_layers_per_virtual_pipeline_stage
    elif pp_size is not None:
        layer_per_chunk = args.num_layers // pp_size
    else:
        layer_per_chunk = args.num_layers

    if vpp_rank is None or not args.enable_recompute_layers_per_pp_rank:
        vpp_rank = 0
    if vpp_size is None or not args.enable_recompute_layers_per_pp_rank:
        vpp_size = 1
    recompute_priority = ((self.layer_number - 1) % layer_per_chunk) * vpp_size + vpp_rank
    full_recompute_layers = args.recompute_num_layers

    if full_recompute_layers:
        if recompute_priority < full_recompute_layers:
            # Do full recomputation
            return False
        elif activation_recompute_layers is None:
            # Do activation function recomputation
            return True
        elif recompute_priority < full_recompute_layers + activation_recompute_layers:
            # Do activation function recomputation
            return True
        else:
            # No recomputation
            return False

    if activation_recompute_layers is None:
        # Do activation function recomputation
        return True
    else:
        return recompute_priority < activation_recompute_layers


def parallel_mlp_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        is_recompute_activation = should_recompute_activation(self)

        def activation_function(*function_args):
            intermediate, bias = function_args

            if self.bias_gelu_fusion:
                assert self.add_bias is True
                assert self.activation_func == F.gelu
                intermediate = bias_gelu_impl(intermediate, bias)
            else:
                if bias_parallel is not None:
                    intermediate = intermediate + bias
                intermediate = self.activation_func(intermediate)
            return intermediate

        if not is_recompute_activation:
            output, output_bias = fn(self, *args, **kwargs)
        else:
            hidden_states = args[0]
            intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)
            self.activation_checkpoint_manager = CheckpointWithoutOutput()
            intermediate_parallel = self.activation_checkpoint_manager.checkpoint(activation_function,
                                                                                  False,
                                                                                  intermediate_parallel,
                                                                                  bias_parallel)
            # [s, b, h]
            output, output_bias = self.dense_4h_to_h(intermediate_parallel)

            # discard the output of the activation function,
            # which will be restored by recomputation during backward.
            self.activation_checkpoint_manager.discard_output()

            # when backward to output of dense_4h_to_h,
            # recompute and restore the output of activation function.
            if output.requires_grad:
                output.register_hook(self.activation_checkpoint_manager.recompute)
        return output, output_bias
    return wrapper


def flash_self_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *arg, **kwargs):
        fn(self, *arg, **kwargs)
        args = get_args()

        self.pse = None
        self.pse_type = args.alibi_fusion_attn_type

        if self.pse_type is None:
            self.pse_type = 1  # not use pse
        elif self.pse_type == 0:
            alibi = AlibiForFusionAttnSingleton.get_alibi_tensor_for_fusion_attn(args.seq_length,
                                                                                 args.num_attention_heads,
                                                                                 args.params_dtype,
                                                                                 args.alibi_diagonal_opposite,
                                                                                 1024)
            self.pse = alibi

        elif self.pse_type == 2 or self.pse_type == 3:
            self.pse = AlibiForFusionAttnSingleton.get_alibi_slopes_for_fusion_attn(args.num_attention_heads)

    return wrapper


def flash_self_attention_forward(self, q, k, v, attention_mask):
    """Implements the multihead softmax attention.
    Arguments
    ---------
        q, k, v: The tensor containing the query, key, and value. (S, B, H, D)
    """
    args = get_args()
    seq_length, _, head_num, head_dim = q.shape[0], q.shape[1], q.shape[2], q.shape[3]

    q, k, v = [rearrange(x, 's b h d -> s b (h d)') for x in [q, k, v]]

    try:
        scale = 1.0 / math.sqrt(head_dim) if self.softmax_scale is None else self.softmax_scale
    except Exception as e:
        raise ValueError('Invalid head_dim: {}'.format(head_dim)) from e

    if args.context_parallel_size > 1 and args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']:
        in_hybrid_mode = False
        if get_context_parallel_group_for_hybrid_ring(check_initialized=False) is not None:
            in_hybrid_mode = True

        if not in_hybrid_mode:
            cp_group = mpu.get_context_parallel_group()
            cp_size = mpu.get_context_parallel_world_size()
            rank = mpu.get_context_parallel_rank()
            cp_global_ranks = mpu.get_context_parallel_global_ranks()
        else:
            cp_group = get_context_parallel_group_for_hybrid_ring()
            cp_size = get_context_parallel_for_hybrid_ring_world_size()
            rank = get_context_parallel_for_hybrid_ring_rank()
            cp_global_ranks = get_context_parallel_for_hybrid_ring_global_ranks()

        cp_para = dict()
        cp_para['causal'] = args.cp_attention_mask_type == 'causal'
        cp_para['cp_group'] = cp_group
        cp_para['cp_size'] = cp_size
        cp_para['rank'] = rank
        cp_para['cp_global_ranks'] = cp_global_ranks
        cp_para['cp_group_for_send_recv_overlap'] = mpu.get_context_parallel_group_for_send_recv_overlap() \
            if args.use_cp_send_recv_overlap else None
        cp_para['pse'] = self.pse
        cp_para['pse_type'] = self.pse_type

        output = ringattn_context_parallel(q, k, v, head_num, cp_para, scale, attention_mask, self.dropout_p)
    else:
        if args.use_fusion_attn_v2:
            output = npu_fusion_attention(
                q, k, v, head_num, args.shape_order,
                pse=self.pse,
                padding_mask=None,
                atten_mask=attention_mask,
                scale=scale,
                pse_type=self.pse_type,
                pre_tokens=args.pre_tockens,
                next_tokens=args.next_tockens,
                keep_prob=1 - self.dropout_p,
                inner_precise=0,
                sparse_mode=args.sparse_mode
            )[0]
        else:
            output = torch_npu.npu_fusion_attention(
                q, k, v, head_num, args.shape_order,
                pse=None,
                padding_mask=None,
                atten_mask=attention_mask,
                scale=scale,
                pre_tockens=args.pre_tockens,
                next_tockens=args.next_tockens,
                keep_prob=1 - self.dropout_p,
                inner_precise=0,
                sparse_mode=args.sparse_mode
            )[0]
    return output


def parallel_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        config = args[0]
        query_projection_size = config.kv_channels * config.num_attention_heads
        _args = get_args()
        if _args.group_query_attention:
            kv_projection_size = _args.kv_channels * _args.num_query_groups
        else:
            kv_projection_size = _args.kv_channels * _args.num_attention_heads
        # qkv bias
        bias = _args.add_qkv_bias or _args.add_bias_linear
        if args[0].context_parallel_size > 1 and args[0].context_parallel_algo in ['ulysses_cp_algo', 'hybrid_cp_algo']:
            ulysses_group = mpu.get_context_parallel_group()
            if args[0].context_parallel_algo == 'hybrid_cp_algo':
                ulysses_group = get_context_parallel_group_for_hybrid_ulysses()
            if self.use_flash_attn:
                self.core_attention_flash = UlyssesContextAttention(self.core_attention_flash, ulysses_group)
            else:
                self.core_attention = UlyssesContextAttention(self.core_attention, ulysses_group)

        if _args.use_nd_matmul:
            self.query_key_value = Nd_ParallelLinear(
                config.hidden_size,
                query_projection_size + 2 * kv_projection_size,
                config=config,
                init_method=config.init_method,
                bias=bias,
                skip_bias_add=True,
                input_is_parallel=True,
                matmul_id=1
            )
        else:
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                query_projection_size + 2 * kv_projection_size,
                config=config,
                init_method=config.init_method,
                bias=bias,
                gather_output=False)

        # dense bias
        bias = _args.add_dense_bias or _args.add_bias_linear
        skip_bias_add = _args.skip_bias_add
        # Output.
        if _args.use_nd_matmul:
            self.dense = Nd_ParallelLinear(
                query_projection_size,
                config.hidden_size,
                config=config,
                init_method=config.output_layer_init_method,
                bias=bias,
                skip_bias_add=True,
                input_is_parallel=True,
                matmul_id=2
            )
        else:
            self.dense = tensor_parallel.RowParallelLinear(
                query_projection_size,
                config.hidden_size,
                config=config,
                init_method=config.output_layer_init_method,
                bias=bias,
                input_is_parallel=True,
                skip_bias_add=skip_bias_add)
        
        if _args.use_nanopipe and parallel_state.get_pipeline_model_parallel_world_size() > 1 \
                    and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            setattr(self.query_key_value, "in_nano", True)
            setattr(self.dense, "in_nano", True)
    return wrapper


def parallel_attention_forward(self, hidden_states, attention_mask,
            encoder_output=None, inference_params=None,
            rotary_pos_emb=None):
    # hidden_states: [sq, b, h]

    # =================================================
    # Pre-allocate memory for key-values for inference.
    # =================================================
    is_first_step = False
    if inference_params:
        if self.layer_number not in inference_params.key_value_memory_dict:
            inf_max_seq_len = inference_params.max_sequence_length
            inf_max_batch_size = inference_params.max_batch_size
            inference_key_memory = self._allocate_memory(
                inf_max_seq_len, inf_max_batch_size,
                self.num_query_groups_per_partition)
            inference_value_memory = self._allocate_memory(
                inf_max_seq_len, inf_max_batch_size,
                self.num_query_groups_per_partition)

            inference_params.key_value_memory_dict[self.layer_number] = (
                inference_key_memory, inference_value_memory)
            is_first_step = True
        else:
            inference_key_memory, inference_value_memory = \
                inference_params.key_value_memory_dict[self.layer_number]

    # =====================
    # Query, Key, and Value
    # =====================
    if self.attention_type == AttnType.self_attn:

        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query_layer,
        key_layer,
        value_layer) = torch.split(
            mixed_x_layer,
            [
                (
                    self.num_attention_heads_per_partition // self.num_query_groups_per_partition
                    * self.hidden_size_per_attention_head
                ),
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head
            ],
            dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn] -
        query_layer = query_layer.view(query_layer.size(0), query_layer.size(1), -1, self.hidden_size_per_attention_head)
    else:
        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        mixed_kv_layer, _ = self.key_value(encoder_output)

        # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
        new_tensor_shape = mixed_kv_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
            2 * self.hidden_size_per_attention_head)
        mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key_layer,
        value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

        # Attention head [sq, b, h] --> [sq, b, hp]
        query_layer, _ = self.query(hidden_states)
        # [sq, b, hp] --> [sq, b, np, hn]
        new_tensor_shape = query_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head)
        query_layer = query_layer.view(*new_tensor_shape)

    # ==================================
    # Adjust key and value for inference
    # ==================================

    # duplicate the pos_emb for self attention
    if rotary_pos_emb is not None:
        if isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = rotary_pos_emb
        else:
            rotary_pos_emb = ((rotary_pos_emb,) * 2)

    if inference_params:
        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + key_layer.size(1)
        assert batch_end <= inference_key_memory.size(1)
        sequence_start = inference_params.sequence_len_offset
        sequence_end = sequence_start + key_layer.size(0)
        assert sequence_end <= inference_key_memory.size(0)
        # Copy key and values.
        inference_key_memory[sequence_start:sequence_end,
                                batch_start:batch_end, ...] = key_layer
        inference_value_memory[sequence_start:sequence_end,
                                batch_start:batch_end, ...] = value_layer
        key_layer = inference_key_memory[
            :sequence_end, batch_start:batch_end, ...]
        value_layer = inference_value_memory[
            :sequence_end, batch_start:batch_end, ...]


        # adjust the key rotary positional embedding
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # need to cross check this condition during inference
            # if not set_inference_key_value_memory:
            if not is_first_step:
                # In inference, we compute one token at a time.
                # Select the correct positional embedding
                # (only the last token in the sequence)
                q_pos_emb = q_pos_emb[sequence_end - 1 : sequence_end]
            else:
                # In the first forward pass of inference,
                # we use the entire provided prefix.
                # q_pos_emb here has the rope embeddings of the entire
                # prefix + to-be-generated output so
                # we slice to just the prefix.
                q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
            k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

    # ==================================
    # core attention computation
    # ==================================

    # expand the key_layer and value_layer [sk, b, ng, hn] -> [sk, b, np, hn]
    if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
        key_layer = key_layer.repeat_interleave(
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
            dim=2
        )
        value_layer = value_layer.repeat_interleave(
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
            dim=2
        )

    # apply relative positional encoding (rotary embedding)
    if rotary_pos_emb is not None:
        q_pos_emb, k_pos_emb = rotary_pos_emb
        query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb, self.config)
        key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb, self.config)

    if not self.use_flash_attn:
        if self.checkpoint_core_attention:
            context_layer = self._checkpointed_attention_forward(
                query_layer, key_layer, value_layer, attention_mask)
        else:
            context_layer = self.core_attention(
                query_layer, key_layer, value_layer, attention_mask)
    else:
        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                context_layer = self.core_attention_flash(query_layer, key_layer, value_layer, attention_mask)
        else:
            context_layer = self.core_attention_flash(query_layer, key_layer, value_layer, attention_mask)

    # =================
    # Output. [sq, b, h]
    # =================

    output, bias = self.dense(context_layer)

    return output, bias


def switch_mlp_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        global_args = get_args()
        if global_args.moe_model_type == 'megatron_moe':
            fn(self, *args, **kwargs)
            return
        from megatron.legacy.model.transformer import SwitchMLP
        super(SwitchMLP, self).__init__()
        config = args[0]
        layer_number = args[1] if len(args) > 1 else None
        from megatron.core.parallel_state import get_expert_model_parallel_group
        from mindspeed.moe.moe import MoE
        from mindspeed.moe.mixtral_parallel_mlpbm import MixtralParallelMLPBM
        try:
            expert_parallel_group = get_expert_model_parallel_group()
        except AttributeError:
            expert_parallel_group = None

        if layer_number is None:
            self.block = MoE(
                global_args.hidden_size,
                MixtralParallelMLPBM(config, ) if global_args.swiglu else ParallelMLP(config, is_expert=False),
                num_experts=global_args.num_experts,
                ep_size=global_args.expert_model_parallel_size,
                k=global_args.moe_router_topk,
                capacity_factor=global_args.moe_train_capacity_factor,
                eval_capacity_factor=global_args.moe_train_capacity_factor,
                aux_loss_coef=global_args.moe_aux_loss_coeff,
                ep_group=expert_parallel_group,
                noisy_gate_policy=global_args.noisy_gate_policy,
                no_drop=global_args.moe_no_drop,
                dynamic_padding=global_args.moe_dynamic_padding,
                use_sinkhorn=global_args.moe_use_sinkhorn,
                sequence_parallel=global_args.sequence_parallel
            )
        else:
            if layer_number % global_args.expert_interval == 0:
                self.block = MoE(
                    global_args.hidden_size,
                    MixtralParallelMLPBM(config, ) if global_args.swiglu else ParallelMLP(config, is_expert=False),
                    num_experts=global_args.num_experts,
                    ep_size=global_args.expert_model_parallel_size,
                    k=global_args.moe_router_topk,
                    capacity_factor=global_args.moe_train_capacity_factor,
                    eval_capacity_factor=global_args.moe_train_capacity_factor,
                    aux_loss_coef=global_args.moe_aux_loss_coeff,
                    ep_group=expert_parallel_group,
                    noisy_gate_policy=global_args.noisy_gate_policy,
                    no_drop=global_args.moe_no_drop,
                    dynamic_padding=global_args.moe_dynamic_padding,
                    use_sinkhorn=global_args.moe_use_sinkhorn,
                    sequence_parallel=global_args.sequence_parallel
                )
            else:
                self.block = ParallelMLP(config)
        return
    return wrapper


def switch_mlp_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        global_args = get_args()
        if global_args.moe_model_type == 'megatron_moe':
            return fn(self, *args, **kwargs)
        hidden_states = args[0]
        used_token = args[1] if len(args) > 1 else None
        output = self.block(hidden_states, used_token)
        return output[0], None
    return wrapper


def parallel_transformer_layer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        from megatron.legacy.model.transformer import SwitchMLP
        super(ParallelTransformerLayer, self).__init__()
        global_args = get_args()
        fn(self, *args, **kwargs)
        if self.mlp.__class__ is SwitchMLP:
            experts_modules = self.mlp.block.moe_layer.experts.experts if global_args.moe_model_type == 'deepspeed_moe' \
                else self.mlp.local_experts
            for expert in experts_modules:
                expert.layer_number = self.layer_number
        else:
            self.mlp.layer_number = self.layer_number

    return wrapper
