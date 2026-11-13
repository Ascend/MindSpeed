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

import torch
from megatron.training import print_rank_0
from megatron.training import get_args
from megatron.core import utils, parallel_state, tensor_parallel


def apply_autopipeline(models):
    if isinstance(models, list):
        for model in models:
            apply_recompute_modules(model)
    else:
        apply_recompute_modules(models)


def apply_recompute_modules(model):
    args = get_args()
    for pp_rankid, recomp_value in enumerate(args.recompute_module_list):
        if pp_rankid == parallel_state.get_pipeline_model_parallel_rank():
            if recomp_value > 0:
                set_recompute_modules(model, recomp_value, args.recompute_type)


def set_recompute_modules(model, recomp_value, module_type):
    recomp_pool = []
    recomp_name = "module.module.language_model.encoder.layers."
    for i in range(0, recomp_value):
        tmp_recomp_name = recomp_name
        tmp_recomp_name += str(i)
        # mlp recompute type
        if module_type == 0:
            tmp_recomp_name += ".mlp"
            recomp_pool.append(tmp_recomp_name)
        # attention recompute type
        if module_type == 1:
            tmp_recomp_name += ".self_attention"
            recomp_pool.append(tmp_recomp_name)
        # layer recompute type
        if module_type == 2:
            recomp_pool.append(tmp_recomp_name)

    for name, module in model.named_modules():
        if name in recomp_pool:
            module.forward = hook_checkpoint_forward(module.forward)


def hook_checkpoint_forward(forward_func):
    def custom_forward(*args, **kargs):
        def inside_forward(*args):
            return forward_func(*args, **kargs)

        return tensor_parallel.checkpoint(inside_forward, None, *args)

    return custom_forward