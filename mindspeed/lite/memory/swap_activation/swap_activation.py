# Copyright 2025 Huawei Technologies Co., Ltd.
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
import logging

import torch

from mindspeed.lite.utils.log import print_rank
from mindspeed.lite.utils.str_match import module_name_match, replace_first_segment_numbers
from mindspeed.lite.memory.swap_activation.tensor_swap_manager import TensorSwapContext
logger = logging.getLogger(__name__)


def swap_activation_modules(model, plan):
    modules = get_swap_activation_modules(model, plan)

    for name, module in modules:
        print_rank(logger.debug, f'[Swap Activation] Applying swap activation to module: <{name}>\n')
        module_tag = replace_first_segment_numbers(name)
        tensor_swap_context = TensorSwapContext(module_tag)
        module.forward = swap_activation_wrapper(module.forward, tensor_swap_context)

    return model


def get_swap_activation_modules(modules, plan):
    matched_modules = []
    for plan_name in plan:
        for name, module in modules.named_modules():
            if module_name_match(plan_name, name):
                matched_modules.append((name, module))
    return matched_modules


def swap_activation_wrapper(function, context, custom_check_fn=None):
    def wrapper(*args, **kwargs):
        hidden_states = None
        if 'hidden_states' in kwargs:
            hidden_states = kwargs['hidden_states']
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = args[0]

        if custom_check_fn is None:
            def default_check_fn(x):
                return x.data_ptr() == hidden_states.data_ptr()
            current_check_fn = default_check_fn
        else:
            current_check_fn = custom_check_fn
        context.set_custom_check_fn(current_check_fn)
        with context:
            return function(*args, **kwargs)

    return wrapper
