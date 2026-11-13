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

from torch.utils.checkpoint import checkpoint

from mindspeed.lite.utils.str_match import module_name_match

logger = logging.getLogger(__name__)


def recompute_modules(model, plan):
    modules = get_recompute_modules(model, plan)

    for name, module in modules:
        print(f'Applying recompute to module: {name}\n')
        module.forward = recompute_wrapper(module.forward)

    return model


def get_recompute_modules(modules, plan):
    matched_modules = []
    for plan_name in plan:
        for name, module in modules.named_modules():
            if module_name_match(plan_name, name):
                matched_modules.append((name, module))
    if len(matched_modules) == 0:
        raise RuntimeError(f'[Recompute] No module named {plan}.')
    return matched_modules


def recompute_wrapper(function):
    def wrapper(*args, **kwargs):
        kwargs['past_key_values'] = None  # transformers kv cache must be set None, or model use_cache=False
        return checkpoint(function, *args, use_reentrant=False, **kwargs)

    return wrapper
