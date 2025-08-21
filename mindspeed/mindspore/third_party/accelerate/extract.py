# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from types import MethodType
import torch
from accelerate.utils.other import is_compiled_module
from accelerate.utils.imports import is_deepspeed_available
from accelerate.utils.transformer_engine import convert_model


def extract_model_from_parallel(
    model, keep_fp32_wrapper: bool = True, keep_torch_compile: bool = True, recursive: bool = False
):
    """
    Extract a model from its distributed containers.

    Args:
        model (`torch.nn.Module`):
            The model to extract.
        keep_fp32_wrapper (`bool`, *optional*):
            Whether to remove mixed precision hooks from the model.
        keep_torch_compile (`bool`, *optional*):
            Whether to unwrap compiled model.
        recursive (`bool`, *optional*, defaults to `False`):
            Whether to recursively extract all cases of `module.module` from `model` as well as unwrap child sublayers
            recursively, not just the top-level distributed containers.

    Returns:
        `torch.nn.Module`: The extracted model.
    """
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)


    is_compiled = is_compiled_module(model)
    if is_compiled:
        compiled_model = model
        model = model._orig_mod

    if is_deepspeed_available():
        from accelerate.utils.deepspeed import DeepSpeedEngine

        options += (DeepSpeedEngine,)

    while isinstance(model, options):
        model = model.module

    if recursive:
        # This is needed in cases such as using FSDPv2 on XLA
        def _recursive_unwrap(module):
            # Wrapped modules are standardly wrapped as `module`, similar to the cases earlier
            # with DDP, DataParallel, DeepSpeed, and FSDP
            if hasattr(module, "module"):
                unwrapped_module = _recursive_unwrap(module.module)
            else:
                unwrapped_module = module
            # Next unwrap child sublayers recursively
            for name, child in unwrapped_module.named_children():
                setattr(unwrapped_module, name, _recursive_unwrap(child))
            return unwrapped_module

        # Start with top-level
        model = _recursive_unwrap(model)

    if not keep_fp32_wrapper:
        forward = model.forward
        original_forward = model.__dict__.pop("_original_forward", None)
        if original_forward is not None:
            while hasattr(forward, "__wrapped__"):
                forward = forward.__wrapped__
                if forward == original_forward:
                    break
            model.forward = MethodType(forward, model)
        if getattr(model, "_converted_to_transformer_engine", False):
            convert_model(model, to_transformer_engine=False)

    if keep_torch_compile and is_compiled:
        compiled_model._orig_mod = model
        model = compiled_model

    return model