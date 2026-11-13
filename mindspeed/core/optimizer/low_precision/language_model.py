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
from megatron.training import get_args


def transformer_language_model_init_wrapper(init_func):
    @wraps(init_func)
    def transformer_language_model_init_func(self, *args, **kwargs):
        init_func(self, *args, **kwargs)
        args = get_args()
        quant_enabled = bool(getattr(args, "quant_states", None))
        if not quant_enabled:
            return
        for name, param in self.named_parameters():
            if "output_layer" in name or "embedding" in name:
                setattr(param, "keep_fp32", True)

    return transformer_language_model_init_func