# Copyright 2026 Huawei Technologies Co., Ltd.
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

from functools import wraps
import torch
from megatron.core.transformer.spec_utils import build_module


def megatron_module_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, config):
        fn(self, config)
        if hasattr(config, 'reset_attention_order') and config.reset_attention_order:
            # Create linear_qkv module before self_attention.
            self.linear_qkv = build_module(torch.nn.GELU)
            # Free memory to avoid memory fragmentation. It will be assigned a real linear function later.
            self.linear_qkv = None
            config.reset_attention_order = False

    return wrapper
