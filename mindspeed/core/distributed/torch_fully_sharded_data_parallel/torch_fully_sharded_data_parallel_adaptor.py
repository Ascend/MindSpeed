"""
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""

from typing import Optional, Set, Iterable, Union
import dataclasses
import importlib

import torch
import yaml

try:
    from torch.distributed import DeviceMesh
    from torch.distributed.fsdp import fully_shard

    HAVE_FSDP = True
except ImportError:
    HAVE_FSDP = False

from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.distributed.fsdp import MixedPrecisionPolicy

from megatron.core.fp8_utils import is_float8tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.distributed import TorchFullyShardedDataParallel

from mindspeed.utils import convert_str_dict_to_real_types, _get_dtype


def _get_class_type(name: str) -> type:
    """
    Args:
        name (str): module.class

    Returns:
        type: Class Type
    """
    names = name.rsplit('.', 1)
    if len(names) == 1:
        raise RuntimeError(f"Please Provide a module.class name, got {name}")
    module_name, class_name = names
    module = importlib.import_module(module_name)
    class_type = getattr(module, class_name, None)
    return class_type


@dataclasses.dataclass
class Fsdp2Config:
    sharding_size: Optional[int] = None
    sub_modules_to_wrap: Optional[Iterable[torch.nn.Module]] = None
    reshard_after_forward: Union[bool, int] = True
    # mp_policy
    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: Optional[torch.dtype] = None
    output_dtype: Optional[torch.dtype] = None
    cast_forward_inputs: bool = True
    # prefetch setting 
    num_to_forward_prefetch: Optional[int] = 0

    ignored_modules: Optional[Iterable[torch.nn.Module]] = None

    def to_dict(self):
        mp_policy = self._mp_policy()
        
        kwargs = {
            "mp_policy": mp_policy,
            "reshard_after_forward": self.reshard_after_forward
        }
        return kwargs

    def _mp_policy(self):
        param_dtype = _get_dtype(self.param_dtype) if self.param_dtype else None
        reduce_dtype = _get_dtype(self.reduce_dtype) if self.reduce_dtype else None
        output_dtype = _get_dtype(self.output_dtype) if self.output_dtype else None
        return MixedPrecisionPolicy(param_dtype=param_dtype,
                                    reduce_dtype=reduce_dtype,
                                    output_dtype=output_dtype,
                                    cast_forward_inputs=self.cast_forward_inputs)

    def _str_to_module(self, module_names: Iterable[str]) -> Set[torch.nn.Module]:
        if module_names:
            try:
                module_set = set(_get_class_type(m_class_name) for m_class_name in module_names)
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(f"Module {module_names} Not Found, \
                                          check yaml config file and your model, or add it to PYTHONPATH") from e
        else:
            module_set = set()
        return module_set 

    @classmethod
    def load_from_yaml(cls, yml_file: str):
        with open(yml_file, 'r') as f:
            config = yaml.safe_load(f)
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name in config:
                kwargs[f.name] = config[f.name]
        return cls(**kwargs)
    
    def __post_init__(self):
        """Post-initialization processing to convert string module names to classes"""
        if self.sub_modules_to_wrap:
            self.sub_modules_to_wrap = self._str_to_module(self.sub_modules_to_wrap)
        if self.ignored_modules:
            self.ignored_modules = self._str_to_module(self.ignored_modules)


def get_fsdp2_mixed_precision_policy(fsdp2_config: dict):
    mp_policy_param_dtype = fsdp2_config.pop('mp_policy_param_dtype', None)
    mp_policy_reduce_dtype = fsdp2_config.pop('mp_policy_reduce_dtype', None)
    mp_policy_output_dtype = fsdp2_config.pop('mp_policy_output_dtype', None)
    mp_policy_cast_forward_inputs = fsdp2_config.pop('mp_policy_cast_forward_inputs', True)
    fsdp2_config['mp_policy'] = MixedPrecisionPolicy(param_dtype=mp_policy_param_dtype,
                                                        reduce_dtype=mp_policy_reduce_dtype,
                                                        output_dtype=mp_policy_output_dtype,
                                                        cast_forward_inputs=mp_policy_cast_forward_inputs)
    return fsdp2_config


def _create_device_mesh(sharding_size: Optional[int], process_group: ProcessGroup) -> DeviceMesh:
    """
    Create a DeviceMesh for FSDP (Fully Sharded Data Parallel).
    
    Args:
        sharding_size (int): Number of processes in each FSDP group (sharding dimension)
        process_group (ProcessGroup): The process group containing all participating ranks
        
    Returns:
        DeviceMesh: A 1D or 2D device mesh for parallel training
    """
    if sharding_size is None:
        sharding_size = torch.distributed.get_world_size(process_group)
    # Get total number of processes in the group
    world_size = torch.distributed.get_world_size(process_group)
    
    # Get global ranks of all processes in this group
    group_global_ranks = torch.tensor(
        get_process_group_ranks(process_group), 
        dtype=torch.int
    )
    
    # Calculate DDP group size (data parallel dimension)
    replicating_size = world_size // sharding_size
    
    # Validate configuration
    if replicating_size * sharding_size != world_size:
        raise ValueError(
            f"World size {world_size} must be divisible by sharding_size {sharding_size}. "
            f"Current configuration would leave {world_size % sharding_size} ranks unassigned."
        )
    
    # Create 1D mesh (FSDP-only) or 2D mesh (FSDP+DDP hybrid)
    if replicating_size == 1:
        # Pure FSDP case - single dimension mesh
        mesh = group_global_ranks
        device_mesh = DeviceMesh(
            "npu",  # NPU device type (change to "cuda" for GPUs)
            mesh, 
            mesh_dim_names=["Shard"]
        )
    else:
        # Hybrid FSDP+DDP case - two dimensional mesh
        mesh = group_global_ranks.view(replicating_size, sharding_size)
        device_mesh = DeviceMesh(
            "npu",
            mesh,
            mesh_dim_names=["Replicate", "Shard"]  # [data_parallel, model_sharding]
        )
    
    return device_mesh


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

    fsdp2_kwargs = {}
    if hasattr(ddp_config, 'fsdp2_config_path'):
        fsdp2_config = Fsdp2Config.load_from_yaml(ddp_config.fsdp2_config_path)
        fsdp2_kwargs.update(fsdp2_config.to_dict())
    
    self.device_mesh = _create_device_mesh(fsdp2_config.sharding_size, self.process_group)
    fsdp2_kwargs["mesh"] = self.device_mesh

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

    sub_modules_to_wrap = sub_modules_to_wrap if fsdp2_config.sub_modules_to_wrap is None else fsdp2_config.sub_modules_to_wrap
    if fsdp2_config.sub_modules_to_wrap is None:
        sub_modules_to_wrap = set(sub_modules_to_wrap)
        for sub_module in self.module.modules():
            fsdp_modules = getattr(sub_module, "_fsdp_modules", [])
            for f in fsdp_modules:
                sub_modules_to_wrap.add(f)
    
    # collect ignored params
    ignored_params = set()
    if fsdp2_config.ignored_modules:
        for sub_module in self.module.modules():
            if isinstance(sub_module, tuple(fsdp2_config.ignored_modules)):
                ignored_params.update(sub_module.parameters())
    fsdp2_kwargs["ignored_params"] = ignored_params

    def _post_order_traverse(module: torch.nn.Module):
        """Post-order traversal of model submodules (recursive implementation).
        
        Yields child modules before their parents.
        """
        for child in module.children():
            yield from _post_order_traverse(child)
        yield module

    prev_module = None
    wrapped_modules_in_order: list[torch.nn.Module] = []
    for sub_module in _post_order_traverse(self.module):
        # Wrap individual submodules to fetch parameters just-in-time rather than
        # conservatively fetching all parameters at the start of each iteration.
        if any(
                isinstance(sub_module, sub_module_to_wrap)
                for sub_module_to_wrap in sub_modules_to_wrap
        ):
            fully_shard(sub_module, **fsdp2_kwargs)

            # Explicitly set the FSDP backward prefetch schedule to prevent activation
            # recomputation from disrupting the automatically generated default schedule.
            if config.recompute_granularity is not None:
                sub_module.set_modules_to_backward_prefetch(
                    [prev_module] if prev_module else []
                )
            prev_module = sub_module
            wrapped_modules_in_order.append(sub_module)

    num_to_forward_prefetch = getattr(fsdp2_config, "num_to_forward_prefetch", 0)
    if num_to_forward_prefetch > 0 and num_to_forward_prefetch < config.num_layers:
        for i, layer in enumerate(wrapped_modules_in_order):
            j_end = min(len(wrapped_modules_in_order), i + 1 + num_to_forward_prefetch)
            layers_to_prefetch = wrapped_modules_in_order[i + 1:j_end]
            if layers_to_prefetch:
                layer.set_modules_to_forward_prefetch(layers_to_prefetch)

    # Wrap the root module as required by the FSDP API.
    fully_shard(self.module, **fsdp2_kwargs)

    restore_custom_attrs(self.module, attrs)