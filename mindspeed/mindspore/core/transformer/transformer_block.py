# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

from functools import wraps
import torch
from torch import Tensor
from megatron.core.transformer.module import MegatronModule


class NoopTransformerLayer(MegatronModule):
    def __init__(self, layer_number):
        super().__init__(None)
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask, context, context_mask, rotary_pos_emb, inference_params, packed_seq_params):
        return hidden_states, context
