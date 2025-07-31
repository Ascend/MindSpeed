# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
import warnings

import torch
from megatron.core import parallel_state, mpu, tensor_parallel
from megatron.core.transformer.module import MegatronModule
from mindspeed.args_utils import get_full_args as get_args
from mindspeed.core.tensor_parallel.checkpoint_manager import get_pipeline_checkpoint_manager

try:
    from einops import rearrange
except ImportError:
    rearrange = None

_GLOBAL_ATTN_MASK = None


def _get_inverted_mask(attention_mask, alibi):
    inverted_mask = attention_mask.to(alibi.dtype)
    inverted_mask = inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), float("-inf")
    )
    return inverted_mask.to(alibi.device) + alibi.unsqueeze(0)


class NoopTransformerLayer(MegatronModule):
    def __init__(self, layer_number):
        super().__init__(None)
        self.layer_number = layer_number

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states.clone()


def set_attention_mask(attn_mask):
    global _GLOBAL_ATTN_MASK
    _GLOBAL_ATTN_MASK = attn_mask


def generate_attention_mask(compress, device):
    global _GLOBAL_ATTN_MASK
    args = get_args()
    if not args.use_flash_attn:
        warnings.warn("Flash Attention is highly recommended")
        _GLOBAL_ATTN_MASK = (torch.tril(torch.ones([args.micro_batch_size, 1, args.seq_length, args.seq_length], dtype=bool, device=device), diagonal=-(args.pre_tockens + 1)) \
                                + torch.triu(torch.ones([args.micro_batch_size, 1, args.seq_length, args.seq_length], dtype=bool, device=device), diagonal=args.next_tockens + 1))
        return

    if compress:
        seq_len = 2048
    else:
        seq_len = args.seq_length
    
    _GLOBAL_ATTN_MASK = torch.triu(
                            torch.ones((seq_len, seq_len), 
                            device=device, dtype=torch.bool), diagonal=1)


def get_attention_mask():
    global _GLOBAL_ATTN_MASK
    if _GLOBAL_ATTN_MASK is not None:
        return _GLOBAL_ATTN_MASK

    args = get_args()
    should_generate_mask = False
    device = 'npu'

    if args.attention_mask_type == 'causal':
        args.sparse_mode = 2
        should_generate_mask = True
        compress = True

    # EoD 模式 Ring Attention的实现
    # general 为基线方案，causal 为加速方案
    # 如果 cp > 1 且使用了Ring Attention 并行（包括Hybrid并行）。则Mask为动态生成的，不需要额外的Mask
    if getattr(args, 'reset_attention_mask', False):
        if args.attention_mask_type == 'general':
            args.sparse_mode = 2
            if args.context_parallel_size == 1 or args.context_parallel_algo == 'ulysses_cp_algo':
                should_generate_mask = True
                compress = True
            else:
                args.sparse_mode = 1
                should_generate_mask = False
        else:
            should_generate_mask = True
            compress = True


    if getattr(args, 'attention_mask_on_cpu', False):
        device = 'cpu'

    if should_generate_mask:
        generate_attention_mask(compress, device)

    return _GLOBAL_ATTN_MASK


def should_recompute(args, layer_number, num_recompute):
    vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    vpp_size = args.virtual_pipeline_model_parallel_size
    pp_size = args.transformer_pipeline_model_parallel_size

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
    recompute_priority = ((layer_number - 1) % layer_per_chunk) * vpp_size + vpp_rank
    full_recompute_layers = args.recompute_num_layers

    if full_recompute_layers:
        if recompute_priority < full_recompute_layers:
            # Do full recomputation
            return False
        elif num_recompute is None:
            return True
        elif recompute_priority < full_recompute_layers + num_recompute:
            return True
        else:
            return False

    if num_recompute is None:
        return True
    else:
        return recompute_priority < num_recompute


def should_recompute_activation(layer_number):
    args = get_args()
    if not args.recompute_activation_function or layer_number is None:
        return False

    if args.recompute_in_bubble or args.recompute_in_advance:
        pipeline_checkpoint_manager = get_pipeline_checkpoint_manager(args.virtual_pipeline_model_parallel_size)
        if pipeline_checkpoint_manager.chunk_do_recompute:
            return False
        elif args.recompute_in_bubble:
            return True

    if args.recompute_activation_function_num_layers is not None:
        if args.recompute_activation_function_num_layers < 0:
            raise AssertionError('--recompute-activation-function-num-layers cannot be less than 0.')
        elif args.recompute_activation_function_num_layers > args.num_layers:
            raise AssertionError('--recompute-activation-function-num-layers cannot be greater than the number of layers.')
    return should_recompute(args, layer_number, args.recompute_activation_function_num_layers)


def should_recompute_norm(self):
    args = get_args()
    if not args.recompute_norm or self.layer_number is None:
        return False
    return should_recompute(args, self.layer_number, args.recompute_norm_num_layers)
