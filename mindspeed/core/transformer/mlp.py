# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from functools import wraps
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.mlp import MLPSubmodules, MLP
from megatron.training import get_args
from mindspeed.core.tensor_parallel.comm_group_api import TPXCollectiveComm, TPXOverlapCollectiveComm, \
    TPYCollectiveComm, TPYOverlapCollectiveComm
from mindspeed.core.tensor_parallel.tp_2d.parallel_linear_2d import ParallelLinear2D


def mlp_init(
    self,
    config: TransformerConfig,
    submodules: MLPSubmodules,
    is_expert: bool = False,
    input_size: int = None,
    shared_expert=False,
):
    super(MLP, self).__init__(config=config)

    self.config: TransformerConfig = config

    self.input_size = input_size if input_size is not None else self.config.hidden_size

    ffn_hidden_size = self.config.ffn_hidden_size
    if self.config.gated_linear_unit:
        ffn_hidden_size *= 2
    if shared_expert:
        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc1',
            shared_expert=shared_expert
        )
    else:
        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc1'
        )

    self.activation_func = self.config.activation_func

    if shared_expert:
        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc2',
            shared_expert=shared_expert
        )
    else:
        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc2'
        )

    self.shared_expert = shared_expert


def mlp_init_2d_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *arg, **kwargs):
        fn(self, *arg, **kwargs)
        args = get_args()
        is_expert = False
        if get_args().num_experts is not None:
            is_expert = True
        if args.tp_2d:
            ffn_hidden_size = self.config.ffn_hidden_size
            if self.config.gated_linear_unit:
                ffn_hidden_size *= 2
            self.linear_fc1 = ParallelLinear2D(
                self.config.hidden_size,
                ffn_hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                add_bias=self.config.add_bias_linear,
                skip_bias_add=True,
                is_expert=is_expert,
                ag_comm_intf=TPXCollectiveComm,
                ag_sd_rcv_overlap_comm_intf=TPXOverlapCollectiveComm,
                rs_comm_intf=TPYCollectiveComm,
                rs_sd_rcv_overlap_comm_intf=TPYOverlapCollectiveComm,
                enable_overlap_ag_with_matmul=False,
                enable_overlap_matmul_with_rs=args.enable_overlap_matmul_with_rs,
                partition_dim=0,
                enable_backward_overlap_ag_with_matmul=args.enable_backward_overlap_ag_with_matmul)
            self.linear_fc2 = ParallelLinear2D(
                self.config.ffn_hidden_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.output_layer_init_method,
                add_bias=self.config.add_bias_linear,
                skip_bias_add=True,
                is_expert=is_expert,
                ag_comm_intf=TPYCollectiveComm,
                ag_sd_rcv_overlap_comm_intf=TPYOverlapCollectiveComm,
                rs_comm_intf=TPXCollectiveComm,
                rs_sd_rcv_overlap_comm_intf=TPXOverlapCollectiveComm,
                enable_overlap_ag_with_matmul=args.enable_overlap_ag_with_matmul,
                enable_overlap_matmul_with_rs=False,
                partition_dim=1,
                enable_backward_overlap_ag_with_matmul=args.enable_backward_overlap_ag_with_matmul)
    return wrapper
