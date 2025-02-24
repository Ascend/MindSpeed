# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
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
import functools
from contextlib import nullcontext
from typing import Optional

import pytest
import torch
from einops import rearrange
from torch import Tensor
from torch.nn import Module

from mindspeed import megatron_adaptor
from mindspeed.core.tensor_parallel.layers import (row_parallel_moe, column_parallel_moe,
                                                   linear_with_grad_accumulation_and_async_allreduce_moe)
from mindspeed.model.transformer import parallel_mlp_forward
from mindspeed.moe.ampipe.ampipe import AttMoEPipe
from mindspeed.moe.ampipe.ampipe_args import ForwardArgs
from mindspeed.patch_utils import MindSpeedPatchesManager as pm
from tests_extend.commons import set_random_seed
from tests_extend.unit_tests.common import DistributedTest

import megatron.core.parallel_state as ps
from megatron.training.arguments import parse_args
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import set_args
from megatron.legacy.model.utils import get_norm
from megatron.legacy.model.transformer import SwitchMLP, ParallelMLP
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.legacy.model.transformer import ParallelAttention
from megatron.legacy.model.enums import AttnType, AttnMaskType
from megatron.core.tensor_parallel.layers import (RowParallelLinear, ColumnParallelLinear,
                                                  linear_with_grad_accumulation_and_async_allreduce)


def get_copied_function(func):
    @functools.wraps(func)
    def copied_function(*args, **kwargs):
        return func(*args, **kwargs)

    return copied_function


ms_parallel_mlp_forward = get_copied_function(parallel_mlp_forward)
ms_row_parallel_forward = get_copied_function(row_parallel_moe)
ms_column_parallel_forward = get_copied_function(column_parallel_moe)
ms_linear_with_grad_accumulation_and_async_allreduce = get_copied_function(
    linear_with_grad_accumulation_and_async_allreduce_moe)

mg_parallel_mlp_forward = get_copied_function(ParallelMLP.forward)
mg_row_parallel_forward = get_copied_function(RowParallelLinear.forward)
mg_column_parallel_forward = get_copied_function(ColumnParallelLinear.forward)
mg_linear_with_grad_accumulation_and_async_allreduce = get_copied_function(
    linear_with_grad_accumulation_and_async_allreduce)


def switch_patch(use_ampipe):
    if use_ampipe:
        pm.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear.forward',
                          ms_row_parallel_forward, True)
        pm.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear.forward',
                          ms_column_parallel_forward, True)
        pm.register_patch('megatron.core.tensor_parallel.layers.linear_with_grad_accumulation_and_async_allreduce',
                          ms_linear_with_grad_accumulation_and_async_allreduce, True)
        pm.register_patch('megatron.legacy.model.transformer.ParallelMLP.forward',
                          ms_parallel_mlp_forward, True)
    else:
        pm.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear.forward',
                          mg_row_parallel_forward, True)
        pm.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear.forward',
                          mg_column_parallel_forward, True)
        pm.register_patch('megatron.core.tensor_parallel.layers.linear_with_grad_accumulation_and_async_allreduce',
                          mg_linear_with_grad_accumulation_and_async_allreduce, True)
        pm.register_patch('megatron.legacy.model.transformer.ParallelMLP.forward',
                          mg_parallel_mlp_forward, True)
    pm.apply_patches()


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Optional[Tensor], Tensor, float, bool) -> Tensor
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x: torch.Tensor,
                                 bias: Optional[torch.Tensor],
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


class TransformerLayerForwardAttnToMlp(Module):
    def __init__(self, config=None):
        super().__init__()
        self.apply_residual_connection_post_norm \
            = config.apply_residual_connection_post_layernorm

        self.input_norm = get_norm(config)
        # Self attention.
        self.self_attention = ParallelAttention(
            config,
            1,
            attention_type=AttnType.self_attn,
            attn_mask_type=AttnMaskType.causal)

        self.hidden_dropout = config.hidden_dropout
        # Normalize the attention output
        self.post_attention_norm = get_norm(config)
        # MLP
        self.mlp = SwitchMLP(config)
        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = \
                nullcontext if use_nvfuser else torch.enable_grad

    def forward(self, hidden_states, attention_mask, ampipe_degree=1):

        # Layer norm at the beginning of the transformer layer.
        norm_output = self.input_norm(hidden_states)
        if ampipe_degree > 1:
            # Self attention.
            q, k, v = self.self_attention(
                norm_output,
                None,
                inference_params=None,
                rotary_pos_emb=None)
            k, v = [rearrange(x, 's b n d -> s b (n d)') for x in [k, v]]
            ampipe_forward_args = ForwardArgs(
                self.self_attention.dense, bias_dropout_add_fused_train, self.post_attention_norm,
                self.mlp.block, self.hidden_dropout
            )
            out_mlp, ln_input = AttMoEPipe.apply(q, k, v, hidden_states, attention_mask, ampipe_forward_args)
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_fused_train(
                    out_mlp,
                    None,
                    ln_input,
                    self.hidden_dropout)

            return output
        # Self attention.
        attention_output, attention_bias = self.self_attention(
                norm_output,
                attention_mask,
                inference_params=None,
                rotary_pos_emb=None)
        # Residual connection.
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = hidden_states

        bias_dropout_add_func = bias_dropout_add_fused_train

        if attention_bias is not None:
            attention_bias = attention_bias.expand_as(residual)
        with self.bias_dropout_add_exec_handler():
            norm_input = bias_dropout_add_func(
                attention_output,
                attention_bias,
                residual,
                self.hidden_dropout)

        # Layer norm post the self attention.
        norm_output = self.post_attention_norm(norm_input)
        # MLP.
        mlp_output, mlp_bias = self.mlp(norm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = norm_input

        if mlp_bias is not None:
            mlp_bias = mlp_bias.expand_as(residual)
        with self.bias_dropout_add_exec_handler():
            output = bias_dropout_add_func(
                mlp_output,
                mlp_bias,
                residual,
                self.hidden_dropout)

        # # Jit compiled function creates 'view' tensor. This tensor
        # # potentially gets saved in the MPU checkpoint function context,
        # # which rejects view tensors. While making a viewless tensor here
        # # won't result in memory savings (like the data loader, or
        # # p2p_communication), it serves to document the origin of this
        # # 'view' tensor.

        return output


class TestAmpipeForward(DistributedTest):
    world_size = 8

    def init_args(self):
        self.args = parse_args(None, True)
        num_attention_heads = 8
        hidden_size = 16
        num_experts = 4
        self.args.num_experts = num_experts
        self.args.moe_router_topk = 2
        self.args.moe_train_capacity_factor = num_experts
        self.args.hidden_size = hidden_size
        self.args.num_attention_heads = num_attention_heads
        self.args.seq_length = 128
        self.args.ampipe_tp_sp_comm_overlap = True
        self.args.use_flash_attn = True
        self.args.sparse_mode = 0
        self.args.use_cp_send_recv_overlap = True
        self.args.attention_dropout = 0.0
        self.args.hidden_dropout = 0.0
        self.args.moe_model_type = 'deepspeed_moe'
        self.args.context_parallel_algo = 'megatron_cp_algo'
        self.args.add_bias_linear = False
        self.args.bias_gelu_fusion = False
        self.args.kv_channels = self.args.hidden_size // self.args.num_attention_heads
        set_args(self.args)

    def compare_ampipe_and_original(self, tp_ep_size, cp_size=1, dtype=torch.float16, ampipe_degree=2):
        self.init_args()
        num_layers = 1
        batch_size = 1
        tp_size, ep_size = tp_ep_size
        self.args.tensor_model_parallel_size = tp_size
        self.args.expert_model_parallel_size = ep_size
        self.args.context_parallel_size = cp_size
        self.args.sequence_parallel = True
        self.args.ampipe_degree = ampipe_degree

        ps.destroy_model_parallel()
        ps.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            expert_model_parallel_size=ep_size)
        set_random_seed(1234)
        model_parallel_cuda_manual_seed(1234)
        config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=self.args.hidden_size,
            num_attention_heads=self.args.num_attention_heads,
            use_cpu_initialization=True,
            fp16=True if dtype == torch.float16 else False,
            bf16=True if dtype == torch.bfloat16 else False,
            hidden_dropout=self.args.hidden_dropout,
            sequence_parallel=self.args.sequence_parallel,
            tensor_model_parallel_size=self.args.tensor_model_parallel_size,
            expert_model_parallel_size=self.args.expert_model_parallel_size,
            add_bias_linear=False,
            num_moe_experts=self.args.num_experts
        )

        b, s = batch_size, self.args.seq_length // tp_size
        hidden_states = torch.randn(s, b, self.args.hidden_size, dtype=dtype, device=torch.npu.current_device())
        attention_mask = torch.triu(
            torch.ones((self.args.seq_length, self.args.seq_length),
                       device=torch.npu.current_device(), dtype=torch.bool),
            diagonal=1)
        if cp_size > 1:
            attention_mask = torch.triu(
                torch.ones((2048, 2048), device=torch.npu.current_device(), dtype=torch.bool),
                diagonal=1
            )

        # baseline ground-truth
        base_transformer = TransformerLayerForwardAttnToMlp(config)
        base_transformer = base_transformer.npu()
        self.args.ampipe_degree = 1
        switch_patch(use_ampipe=False)
        hidden_states_base = hidden_states.clone().detach()
        hidden_states_base.requires_grad = True

        out = base_transformer(hidden_states_base, attention_mask)
        # baseline backward
        out.sum().backward()
        hidden_states_base_grad = hidden_states_base.grad

        # ampipe
        self.args.ampipe_degree = ampipe_degree
        self.args.ampipe_tp_sp_comm_overlap = True
        switch_patch(use_ampipe=True)
        hidden_states_ampipe = hidden_states.clone().detach()
        hidden_states_ampipe.requires_grad = True
        out_ampipe = base_transformer(hidden_states_ampipe, attention_mask, ampipe_degree)
        out_ampipe.sum().backward()
        hidden_states_ampipe_grad = hidden_states_ampipe.grad

        # same as transformer_engine
        tols = dict(atol=5e-3, rtol=5e-3)
        if dtype == torch.bfloat16:
            tols = dict(atol=2.5e-2, rtol=2.5e-2)

        # compare forward results with and without ampipe
        assert torch.allclose(out, out_ampipe, **tols)

        # compare backward results with and without ampipe
        assert torch.allclose(hidden_states_base_grad, hidden_states_ampipe_grad, **tols)

    @pytest.mark.skip(reason='Because DeepSpeed MoE was removed, the corresponding test case has been skipped.')
    @pytest.mark.parametrize("tp_ep", [
            (2, 2),
            (2, 4),
            (4, 2),
            (8, 1),
        ])
    def test_ampipe_sp_no_cp_bf16(self, tp_ep):
        self.compare_ampipe_and_original(tp_ep, 1, dtype=torch.bfloat16, ampipe_degree=2)

    @pytest.mark.skip(reason='Because DeepSpeed MoE was removed, the corresponding test case has been skipped.')
    @pytest.mark.parametrize("tp_ep", [
            (2, 2),
            (2, 4),
            (4, 2),
        ])
    def test_ampipe_sp_cp_bf16(self, tp_ep):
        self.compare_ampipe_and_original(tp_ep, 2, dtype=torch.bfloat16, ampipe_degree=2)
