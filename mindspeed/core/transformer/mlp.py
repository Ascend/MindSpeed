# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from functools import wraps
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.mlp import MLPSubmodules, MLP


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

    self.activation_func = self.config.activation_func

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

    self.shared_expert = shared_expert