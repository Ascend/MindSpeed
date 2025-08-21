# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import os
import re
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union
from zipfile import is_zipfile
from packaging import version

import torch
from torch import nn, Tensor
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled, is_local_dist_rank_0, _load_parameter_into_model, _infer_parameter_dtype, PreTrainedModel, ModuleUtilsMixin
from transformers.quantizers.base import HfQuantizer
from transformers.quantizers.quantizers_utils import get_module_from_name
from transformers.utils.import_utils import ENV_VARS_TRUE_VALUES, is_safetensors_available, is_torch_xla_available
from transformers.utils.quantization_config import QuantizationMethod
from transformers.integrations.tensor_parallel import shard_and_distribute_module
from accelerate.utils.offload import offload_weight

XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0").upper()
XLA_DOWNCAST_BF16 = os.environ.get("XLA_DOWNCAST_BF16", "0").upper()


@contextmanager
def safe_open(file: str, framework: str = "pt", device: str = "cpu"):
    class SafeTensorFile:
        def __init__(self, framework: str):
            self._framework = framework

        def metadata(self) -> dict:
            return {"format": self._framework}

    try:
        file = SafeTensorFile(framework=framework)
        yield file
    finally:
        pass


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike],
    is_quantized: bool = False,
    map_location: Optional[Union[str, torch.device]] = "cpu",
    weights_only: bool = True,
):
    """
    Reads a `safetensor` or a `.bin` checkpoint file. We load the checkpoint on "cpu" by default.
    """
    if checkpoint_file.endswith(".safetensors") and is_safetensors_available():
        from torch.serialization import safe_load_file
        return safe_load_file(checkpoint_file)

    try:
        if map_location is None:
            if (
                    (
                            is_deepspeed_zero3_enabled()
                            and torch.distributed.is_initialized()
                            and torch.distributed.get_rank() > 0
                    )
                    or (is_fsdp_enabled() and not is_local_dist_rank_0())
            ) and not is_quantized:
                map_location = "meta"
            else:
                map_location = "cpu"
        extra_args = {}
        # mmap can only be used with files serialized with zipfile-based format.
        if (
                isinstance(checkpoint_file, str)
                and map_location != "meta"
                and version.parse(torch.__version__) >= version.parse("2.1.0")
                and is_zipfile(checkpoint_file)
        ):
            extra_args = {"mmap": True}
        return torch.load(
            checkpoint_file,
            map_location=map_location,
            weights_only=weights_only,
            **extra_args,
        )
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read(7) == "version":
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from pytorch checkpoint file for '{checkpoint_file}' "
                f"at '{checkpoint_file}'. "
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True."
            )


@torch.no_grad()
def _load_state_dict_into_meta_model(
    model: "PreTrainedModel",
    state_dict: Dict,
    shard_file: str,
    expected_keys: List[str],
    reverse_renaming_mapping: Dict[str, str],
    device_map: Optional[Dict] = None,
    disk_offload_folder: Optional[str] = None,
    disk_offload_index: Optional[Dict] = None,
    cpu_offload_folder: Optional[str] = None,
    cpu_offload_index: Optional[Dict] = None,
    hf_quantizer: Optional[HfQuantizer] = None,
    is_safetensors: bool = False,
    keep_in_fp32_regex: Optional[re.Pattern] = None,
    unexpected_keys: Optional[List[str]] = None,  # passing `unexpected` for cleanup from quantization items
    device_mesh: Optional["torch.distributed.device_mesh.DeviceMesh"] = None,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Load parameters from `meta_state_dict` into the model. The parameters of the `meta_state_dict` are on the meta
    device in order to easily infer the shapes and dtypes that they will have. Then proper parameters are then loaded
    from `shard_file`, which is the actual state dict file on disk.
    This function takes care of correctly casting dtypes, devices, and sharding tensors in case of tensor parallelism.
    """
    tensor_device = "cpu"
    if device_map is not None and device_map.get("", None) is not None:
        if device_map[""] not in ("cpu", torch.device("cpu")):
            tensor_device = device_map[""].index if isinstance(device_map[""], torch.device) else device_map[""]
    if device_map is not None:
        device_map_regex = "|".join([re.escape(k) for k in sorted(device_map.keys(), reverse=True)])

    is_quantized = hf_quantizer is not None
    is_hqq_or_bnb = is_quantized and hf_quantizer.quantization_config.quant_method in [
        QuantizationMethod.HQQ,
        QuantizationMethod.BITS_AND_BYTES,
    ]
    is_meta_state_dict = shard_file.endswith(".safetensors") and not is_hqq_or_bnb
    file_pointer = None
    if is_meta_state_dict:
        from torch.serialization import safe_load_file
        file_pointer = safe_load_file(shard_file)
    for param_name, empty_param in state_dict.items():
        if param_name not in expected_keys:
            continue

        # we need to use serialized_param_name as file pointer is untouched
        if is_meta_state_dict:
            # This is the name of the parameter as it appears on disk file
            serialized_param_name = reverse_renaming_mapping[param_name]
            param = file_pointer[serialized_param_name]
        else:
            param = empty_param.to(tensor_device)  # It is actually not empty!

        to_contiguous, casting_dtype = _infer_parameter_dtype(
            model,
            param_name,
            empty_param,
            keep_in_fp32_regex,
            hf_quantizer,
        )

        if device_mesh is not None:  # In this case, the param is already on the correct device!
            shard_and_distribute_module(
                model,
                param,
                empty_param,
                param_name,
                casting_dtype,
                to_contiguous,
                int(os.environ["RANK"]),  # the rank
                device_mesh,
            )
        else:
            param = param[...]
            if casting_dtype is not None:
                param = param.to(casting_dtype)
            if to_contiguous:
                param = param.contiguous()

            if device_map is None:
                param_device = "cpu"
            else:
                module_layer = re.search(device_map_regex, param_name)
                if not module_layer:
                    raise ValueError(f"{param_name} doesn't have any device set.")
                else:
                    param_device = device_map[module_layer.group()]

            if param_device == "disk":
                if not is_safetensors:
                    disk_offload_index = offload_weight(param, param_name, disk_offload_folder, disk_offload_index)
            elif param_device == "cpu" and cpu_offload_index is not None:
                cpu_offload_index = offload_weight(param, param_name, cpu_offload_folder, cpu_offload_index)
            elif (
                not is_quantized
                or (not hf_quantizer.requires_parameters_quantization)
                or (
                    not hf_quantizer.check_quantized_param(
                        model,
                        param,
                        param_name,
                        state_dict,
                        param_device=param_device,
                        device_map=device_map,
                    )
                )
            ):
                if is_fsdp_enabled():
                    param_device = "cpu" if is_local_dist_rank_0() else "meta"

                _load_parameter_into_model(model, param_name, param.to(param_device))

            else:
                hf_quantizer.create_quantized_param(
                    model, param, param_name, param_device, state_dict, unexpected_keys
                )
                # For quantized modules with FSDP/DeepSpeed Stage 3, we need to quantize the parameter on the GPU
                # and then cast it to CPU to avoid excessive memory usage on each GPU
                # in comparison to the sharded model across GPUs.
                if is_fsdp_enabled() or is_deepspeed_zero3_enabled():
                    module, param_type = get_module_from_name(model, param_name)
                    value = getattr(module, param_type)
                    param_to = "cpu"
                    if is_fsdp_enabled() and not is_local_dist_rank_0():
                        param_to = "meta"
                    val_kwargs = {}
                    if hasattr(module, "weight") and module.weight.__class__.__name__ == "Int8Params":
                        val_kwargs["requires_grad"] = False
                    value = type(value)(value.data.to(param_to), **val_kwargs, **value.__dict__)
                    setattr(module, param_type, value)

    return disk_offload_index, cpu_offload_index


def get_parameter_dtype(parameter: Union[nn.Module, "ModuleUtilsMixin"]):
    class DtypeWrapper:
        def __init__(self, dtype):
            self._dtype = dtype

        def __getattr__(self, name):
            return getattr(self._dtype, name)

        def __eq__(self, other):
            return self._dtype == other

        def __str__(self):
            return f"torch.{str(self._dtype).replace('torch.', '').lower()}"

        def __repr__(self):
            return str(self)

    def wrap(dtype):
        if dtype is None:
            return None
        return DtypeWrapper(dtype)

    last_dtype = None
    for t in parameter.parameters():
        last_dtype = t.dtype
        if t.is_floating_point():
            if XLA_USE_BF16 in ENV_VARS_TRUE_VALUES and is_torch_xla_available():
                return wrap(torch.bfloat16)
            if XLA_DOWNCAST_BF16 in ENV_VARS_TRUE_VALUES and is_torch_xla_available():
                if t.dtype == torch.float:
                    return wrap(torch.bfloat16)
                if t.dtype == torch.double:
                    return wrap(torch.float32)
            return wrap(t.dtype)

    if last_dtype is not None:
        return wrap(last_dtype)

    def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
        return [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]

    gen = parameter._named_members(get_members_fn=find_tensor_attributes)
    last_tuple = None
    for tuple_item in gen:
        last_tuple = tuple_item
        if tuple_item[1].is_floating_point():
            return wrap(tuple_item[1].dtype)

    if last_tuple is not None:
        return wrap(last_tuple[1].dtype)

    for t in parameter.buffers():
        last_dtype = t.dtype
        if t.is_floating_point():
            return wrap(t.dtype)
    return wrap(last_dtype)