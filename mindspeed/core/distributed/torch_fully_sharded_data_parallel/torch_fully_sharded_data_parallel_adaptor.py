"""
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""

from typing import Optional, Set

import torch

try:
    from torch.distributed import DeviceMesh
    from torch.distributed.fsdp import fully_shard

    HAVE_FSDP = True
except ImportError:
    HAVE_FSDP = False

from torch.distributed import ProcessGroup
from torch.distributed.fsdp import MixedPrecisionPolicy

from megatron.core.fp8_utils import is_float8tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.distributed import TorchFullyShardedDataParallel

from mindspeed.utils import convert_str_dict_to_real_types


def get_fsdp2_config_attrs(fsdp2_config_str):
    config_dict = {}

    # split key-value pairs, data example: key1=value1;key2=value2
    pairs = [pair.strip() for pair in fsdp2_config_str.split(';') if pair.strip()]

    for pair in pairs:
        key, value = pair.split('=', 1)
        key = key.strip()
        value = value.strip()

        convert_str_dict_to_real_types(config_dict, key, value)

    return config_dict


def get_fsdp2_mixed_precision_policy(fsdp2_config: dict):
    mp_policy_param_dtype = fsdp2_config.pop('mp_policy_param_dtype', None)
    mp_policy_reduce_dtype = fsdp2_config.pop('mp_policy_reduce_dtype', None)
    mp_policy_output_dtype = fsdp2_config.pop('mp_policy_output_dtype', None)
    mp_policy_cast_forward_inputs = fsdp2_config.pop('mp_policy_cast_forward_inputs', False)
    fsdp2_config['mp_policy'] = MixedPrecisionPolicy(param_dtype=mp_policy_param_dtype,
                                                        reduce_dtype=mp_policy_reduce_dtype,
                                                        output_dtype=mp_policy_output_dtype,
                                                        cast_forward_inputs=mp_policy_cast_forward_inputs)
    return fsdp2_config


def torch_fully_sharded_data_parallel_init(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        disable_bucketing: bool = False,
        sub_modules_to_wrap: Set[torch.nn.Module] = {
            TransformerLayer,
            LanguageModelEmbedding,
            RotaryEmbedding,
            tensor_parallel.ColumnParallelLinear,
        },
        process_group: Optional[ProcessGroup] = None,
):
    assert (
        HAVE_FSDP
    ), 'TorchFullyShardedDataParallel requires PyTorch >= 2.4.0 with FSDP 2 support.'

    super(TorchFullyShardedDataParallel, self).__init__(config=config, module=module)

    if process_group is None:
        self.process_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    else:
        self.process_group = process_group

    self.device_mesh = DeviceMesh.from_group(self.process_group, "npu")
    kwargs = {
        "mesh": self.device_mesh,
        "reshard_after_forward": getattr(ddp_config, "reshard_after_forward", True),
    }

    # manually convert str to dist configuration
    if hasattr(ddp_config, 'fsdp2_config_str'):
        fsdp2_config = get_fsdp2_config_attrs(ddp_config.fsdp2_config_str)
        fsdp2_config = get_fsdp2_mixed_precision_policy(fsdp2_config)
        kwargs.update(fsdp2_config)

    self.ddp_config = ddp_config

    def save_custom_attrs(module):
        custom_attrs = {}
        for name, param in module.named_parameters():
            attrs = vars(param)
            if is_float8tensor(param):
                # disable fp8 transpose cache and perform transposing fp8 weights
                # at each micro-batch because torch-FSDP doesn't recognize the
                # micro-batch id, thus removing unnecessary memory stores
                attrs['_fp8_attrs']['transpose_invalid'] = False
                del attrs['_fp8_attrs']['transpose']
            custom_attrs[name] = {k: v for k, v in attrs.items()}
        return custom_attrs

    def restore_custom_attrs(module, custom_attrs):
        for name, param in module.named_parameters():
            if name in custom_attrs:
                for attr_name, attr_value in custom_attrs[name].items():
                    setattr(param, attr_name, attr_value)

    # Save the custom attributes on Parameters before FSDP overwrites them.
    attrs = save_custom_attrs(self.module)

    sub_modules_to_wrap = set(sub_modules_to_wrap)
    for sub_module in self.module.modules():
        fsdp_modules = getattr(sub_module, "_fsdp_modules", [])
        for f in fsdp_modules:
            sub_modules_to_wrap.add(f)

    prev_module = None
    for sub_module in reversed(list(self.module.modules())):
        # Wrap individual submodules to fetch parameters just-in-time rather than
        # conservatively fetching all parameters at the start of each iteration.
        if any(
                isinstance(sub_module, sub_module_to_wrap)
                for sub_module_to_wrap in sub_modules_to_wrap
        ):
            fully_shard(sub_module, **kwargs)

            # Explicitly set the FSDP backward prefetch schedule to prevent activation
            # recomputation from disrupting the automatically generated default schedule.
            if config.recompute_granularity is not None:
                sub_module.set_modules_to_backward_prefetch(
                    [prev_module] if prev_module else []
                )
            prev_module = sub_module

    # Wrap the root module as required by the FSDP API.
    fully_shard(self.module, **kwargs)

    restore_custom_attrs(self.module, attrs)