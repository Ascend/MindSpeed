# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

import gc
import dataclasses
import importlib
from functools import wraps
from typing import Tuple, Literal, Union, Iterable, Optional

import yaml
import torch
import torch.nn as nn
import torch.distributed as dist

from megatron.core import mpu
from megatron.core.optimizer import OptimizerConfig
from megatron.training.training import get_optimizer_param_scheduler, get_model
from megatron.training.global_vars import get_args, get_timers
from megatron.training.utils import (
    print_rank_0,
    unwrap_model,
)
from megatron.core.utils import get_model_config
from megatron.training.checkpointing import load_checkpoint

from mindspeed.core.distributed.layerzero.zero3 import LayerZeRO3
from mindspeed.core.distributed.layerzero.zero3.wrap import ModuleWrapPolicy
from mindspeed.core.distributed.layerzero.zero3.api import (
    BackwardPrefetch,
    BackwardReduceScatter,
    MixedPrecision,
)
from mindspeed.core.distributed.layerzero.megatron_adaptor import get_optimizer
from mindspeed.core.distributed.layerzero.state.mga_checkpoint import save_checkpoint, load_layerzero_checkpoint
from . import constants
#!===============Globals============================
_ZERO1_PROCESS_GROUP = None
_ZERO3_PROCESS_GROUP = None
_ZERO1_PROCESS_GROUP_RANKS = None
_ZERO3_PROCESS_GROUP_RANKS = None
_TP_ZERO1_PROCESS_GROUP = None
_TP_ZERO1_PROCESS_GROUP_RANKS = None
_TP_ZERO3_PROCESS_GROUP = None
_TP_ZERO3_PROCESS_GROUP_RANKS = None


@dataclasses.dataclass
class LayerzeroConfig:
    zero3_size: int = 8
    transformer_layers:  Optional[Iterable[torch.nn.Module]] = None
    backward_prefetch: Literal["BACKWARD_PRE",
                               "BACKWARD_POST"] = 'BACKWARD_PRE'
    backward_reduce_scatter: Literal["BACKWARD_PRE",
                                     "BACKWARD_POST"] = 'BACKWARD_PRE'
    param_dtype: Optional[Literal["fp16", "bf16", "fp32"]] = "fp16"
    reduce_dtype:  Optional[Literal["fp16", "bf16", "fp32"]] = "fp16"
    buffer_dtype:  Optional[Literal["fp16", "bf16", "fp32"]] = None
    ignored_modules:  Optional[Iterable[torch.nn.Module]] = None
    param_init_fn: Optional[str] = None,
    forward_prefetch: bool = True
    limit_all_gathers: bool = True
    offload_grads: bool = False
    ckpt_load_path: str = None
    autocast_input: bool = True
    autocast_output: bool = True
    
    def __post_init__(self):
        if self.zero3_size <= 0 or not isinstance(self.zero3_size, int):
            raise ValueError("zero3_size must be a non-negative int value")    
            
    @classmethod
    def load_from_yaml(cls, yml_file: str):
        with open(yml_file, 'r') as f:
            config = yaml.safe_load(f)
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name in config:
                kwargs[f.name] = config[f.name]
        print_rank_0(kwargs)
        return cls(**kwargs)

    def to_dict(self):
        process_group = self._process_group()
        wrap_policy = self._wrap_policy()
        mixed_precision = self._mp_policy()
        backward_prefetch = self._backward_prefetch()
        backward_rs = self._backward_reduce_scatter()
        kwargs = {
            "process_group": process_group,
            "tp_zero_process_group": self._tp_process_group(),
            "auto_wrap_policy": wrap_policy,
            "mixed_precision": mixed_precision,
            "device_id": torch.cuda.current_device(),
            "backward_prefetch": backward_prefetch,
            "backward_reduce_scatter": backward_rs,
            "forward_prefetch": self.forward_prefetch,
            "offload_grads": self.offload_grads
        }
        return kwargs

    def _mp_policy(self):
        # if self.fwd_bwd_dtype or
        param_dtype = _get_dtype(
            self.param_dtype) if self.param_dtype else None
        reduce_dtype = _get_dtype(
            self.reduce_dtype) if self.reduce_dtype else None
        buffer_dtype = _get_dtype(
            self.buffer_dtype) if self.buffer_dtype else None
        return MixedPrecision(param_dtype=param_dtype,
                              reduce_dtype=reduce_dtype,
                              buffer_dtype=buffer_dtype)

    def _wrap_policy(self):
        if self.transformer_layers:
            try:
                transformer_layer_cls = set(_get_class_type(
                    m_class_name) for m_class_name in self.transformer_layers)
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(f"Module {transformer_layer_cls} Not Found, \
                                          check yaml config file and your model, or add it to PYTHONPATH") from e
        else:
            transformer_layer_cls = []
        print_rank_0(f"Each of these layers will be wrapped as a single layer:{transformer_layer_cls}")
        wrap_policy = ModuleWrapPolicy(transformer_layer_cls)
        return wrap_policy

    def _process_group(self):
        if not _is_layerzero_pg_initialized():
            raise RuntimeError("Layerzero process group is not initialized")
        return _ZERO3_PROCESS_GROUP, _ZERO1_PROCESS_GROUP

    def _tp_process_group(self):
        return _TP_ZERO3_PROCESS_GROUP, _TP_ZERO1_PROCESS_GROUP

    def _backward_prefetch(self):
        if self.backward_prefetch not in ['BACKWARD_PRE', 'BACKWARD_POST']:
            raise ValueError(f"{self.backward_prefetch} is not supported")
        return BackwardPrefetch[self.backward_prefetch]

    def _backward_reduce_scatter(self):
        if self.backward_reduce_scatter not in ['BACKWARD_PRE', 'BACKWARD_POST']:
            raise ValueError(f"{self.backward_reduce_scatter} is not supported")
        return BackwardReduceScatter[self.backward_reduce_scatter]

    def setup_cast_settings(self):
        constants.set_auto_cast_input(self.autocast_input)
        constants.set_auto_cast_output(self.autocast_output)


def _get_module_attr(model: nn.Module, name: Iterable[str]):
    if name is None:
        return None
    if not isinstance(name, list):
        name = [name]
    name = set(list(name))
    if not all(isinstance(n, str) for n in name):
        raise AssertionError("All name should be str")
    results = set(getattr(model, n, None) for n in name)
    if all([m is None for m in results]):
        return None
    return results


def _get_module_and_class(name: str) -> Tuple[str, str]:
    names = name.rsplit('.', 1)
    if len(names) == 1:
        raise RuntimeError(f"Please Provide a module.class name, got {name}")
    module_name, class_name = names
    return module_name, class_name


def _get_class_type(name: str) -> type:
    """
    Args:
        name (str): module.class

    Returns:
        type: Class Type
    """
    module_name, class_name = _get_module_and_class(name)
    module = importlib.import_module(module_name)
    class_type = getattr(module, class_name, None)
    return class_type


def _get_dtype(dtype: str):
    if dtype not in {'fp16', 'bf16', 'fp32'}:
        raise AssertionError(f"dtype {dtype} not Supported")
    if dtype == 'fp16':
        return torch.float16
    elif dtype == 'bf16':
        return torch.bfloat16
    elif dtype == 'fp32':
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def wrap_model_with_layerzero(model: Union[Iterable[torch.nn.Module], torch.nn.Module], lz_config: LayerzeroConfig):

    kwargs = lz_config.to_dict()
    if isinstance(model, nn.Module):
        model = [model]

    model_list = []
    for model_chunk in model:
        ignored_modules = _get_module_attr(
            model_chunk, lz_config.ignored_modules)
        kwargs["ignored_modules"] = ignored_modules
        zero3_model = LayerZeRO3(model_chunk, **kwargs)
        model_list.append(zero3_model)
    return model_list


def create_optimizer_layerzero(model,
                               no_wd_decay_cond=None,
                               scale_lr_cond=None,
                               lr_mult=1.0):
    args = get_args()
    timers = get_timers()
    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = timers
    optimizer = get_optimizer(config, model[0], no_wd_decay_cond,
                              scale_lr_cond, lr_mult)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
    return optimizer, opt_param_scheduler


def layerzero_setup_model_and_optimizer_wrapper(setup_model_and_optimizer):
    @wraps(setup_model_and_optimizer)
    def wrapper(model_provider_func,
                model_type,
                no_wd_decay_cond=None,
                scale_lr_cond=None,
                lr_mult=1.0):
        args = get_args()
        if getattr(args, 'layerzero', False):
            # ========================================================
            timers = get_timers()
            models = get_model(model_provider_func, model_type, False)
            if args.load is not None or args.pretrained_checkpoint is not None:
                timers('load-checkpoint', log_level=0).start(barrier=True)
                args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                    models, None, None)
                timers('load-checkpoint').stop(barrier=True)
                timers.log(['load-checkpoint'])
            else:
                args.iteration = 0
                args.num_floating_point_operations_so_far = 0
            # ========================================================
            config_yaml = args.layerzero_config
            config = LayerzeroConfig.load_from_yaml(config_yaml)
            config.setup_cast_settings()
            zero_models = wrap_model_with_layerzero(
                unwrap_model(models), config)
            del models
            gc.collect()

            optimizer, opt_param_scheduler = create_optimizer_layerzero(zero_models,
                                                                        no_wd_decay_cond=no_wd_decay_cond,
                                                                        scale_lr_cond=scale_lr_cond,
                                                                        lr_mult=lr_mult)
            if config.ckpt_load_path is not None:
                load_layerzero_checkpoint(
                    zero_models, config.ckpt_load_path, optimizer, opt_param_scheduler)
            torch.cuda.empty_cache()
            print_rank_0(f"{zero_models[0]=}")

            model_config = get_model_config(zero_models[0])
            if len(zero_models) == 1:
                model_config.no_sync_func = zero_models[0].no_sync
            else:
                model_config.no_sync_func = [m.no_sync for m in zero_models]
            return zero_models, optimizer, opt_param_scheduler
        else:
            return setup_model_and_optimizer(model_provider_func,
                                             model_type,
                                             no_wd_decay_cond,
                                             scale_lr_cond,
                                             lr_mult)

    return wrapper


def initialize_zero_process_group_with_pp(pp_size, zero3_size):
    global _ZERO1_PROCESS_GROUP
    global _ZERO1_PROCESS_GROUP_RANKS
    global _ZERO3_PROCESS_GROUP
    global _ZERO3_PROCESS_GROUP_RANKS

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    zero1_size = world_size // pp_size
    zero3_size = min(zero3_size, zero1_size)
    ensure_divisibility(zero1_size, zero3_size)
    num_zero3_groups = zero1_size // zero3_size

    for zero1_idx in range(pp_size):
        cur_zero1_ranks = list(
            range(zero1_idx * zero1_size, (zero1_idx + 1) * zero1_size))
        zero1_group = dist.new_group(ranks=cur_zero1_ranks, backend="hccl")
        if global_rank in cur_zero1_ranks:
            _ZERO1_PROCESS_GROUP = zero1_group
            _ZERO1_PROCESS_GROUP_RANKS = cur_zero1_ranks

        for zero3_idx in range(num_zero3_groups):
            cur_zero3_ranks = cur_zero1_ranks[zero3_idx *
                                              zero3_size: (zero3_idx + 1) * zero3_size]
            zero3_group = dist.new_group(ranks=cur_zero3_ranks, backend="hccl")
            if global_rank in cur_zero3_ranks:
                _ZERO3_PROCESS_GROUP = zero3_group
                _ZERO3_PROCESS_GROUP_RANKS = cur_zero3_ranks
    return


def initialize_tp_zero_process_group(tp_zero3_size: int):
    if not mpu.is_initialized() or not _is_layerzero_pg_initialized():
        raise RuntimeError("Mpu or ZeRO process group is not initialized")

    global _TP_ZERO1_PROCESS_GROUP
    global _TP_ZERO1_PROCESS_GROUP_RANKS
    global _TP_ZERO3_PROCESS_GROUP
    global _TP_ZERO3_PROCESS_GROUP_RANKS

    _TP_ZERO1_PROCESS_GROUP = mpu.get_data_parallel_group(
        with_context_parallel=True)
    _TP_ZERO1_PROCESS_GROUP_RANKS = list(
        mpu._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP)
    tp_zero1_size = len(_TP_ZERO1_PROCESS_GROUP_RANKS)
    tp_zero3_size = min(tp_zero1_size, tp_zero3_size)
    ensure_divisibility(tp_zero1_size, tp_zero3_size)

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    num_zero1_groups = world_size // tp_zero1_size
    num_zero3_groups = tp_zero1_size // tp_zero3_size
    for zero1_idx in range(num_zero1_groups):
        for zero3_idx in range(num_zero3_groups):
            cur_zero1_ranks = list(
                range(zero1_idx, world_size, num_zero1_groups))
            group_ranks = cur_zero1_ranks[zero3_idx *
                                          tp_zero3_size: (zero3_idx + 1) * tp_zero3_size]
            group = dist.new_group(ranks=group_ranks, backend="hccl")
            if global_rank in group_ranks:
                _TP_ZERO3_PROCESS_GROUP = group
                _TP_ZERO3_PROCESS_GROUP_RANKS = group_ranks
    return


def initialized_zero_process_group(zero3_size):
    '''
    For TP > 1 or PP > 1 or TP + PP situation, the process group needs to be taken care of.
    '''
    if not mpu.is_initialized():
        raise AssertionError(f"mpu is not initialized")
    args = get_args()
    global _ZERO1_PROCESS_GROUP
    global _ZERO1_PROCESS_GROUP_RANKS
    global _ZERO3_PROCESS_GROUP
    global _ZERO3_PROCESS_GROUP_RANKS
    global _TP_ZERO1_PROCESS_GROUP
    global _TP_ZERO1_PROCESS_GROUP_RANKS
    global _TP_ZERO3_PROCESS_GROUP
    global _TP_ZERO3_PROCESS_GROUP_RANKS

    initialize_zero_process_group_with_pp(
        args.pipeline_model_parallel_size, zero3_size)
    #! process TP process groups
    if args.tensor_model_parallel_size > 1:
        ensure_divisibility(zero3_size, args.tensor_model_parallel_size)
        tp_zero3_size = max(1, zero3_size // args.tensor_model_parallel_size)
        initialize_tp_zero_process_group(tp_zero3_size)
    else:
        _TP_ZERO1_PROCESS_GROUP = _ZERO1_PROCESS_GROUP
        _TP_ZERO1_PROCESS_GROUP_RANKS = _ZERO1_PROCESS_GROUP_RANKS
        _TP_ZERO3_PROCESS_GROUP = _ZERO3_PROCESS_GROUP
        _TP_ZERO3_PROCESS_GROUP_RANKS = _ZERO3_PROCESS_GROUP_RANKS

    print(f"Layerzero with zero1 process group: {_ZERO1_PROCESS_GROUP_RANKS}, \
            zero3 process group: {_ZERO3_PROCESS_GROUP_RANKS}, \
            TP zero1 process group: {_TP_ZERO1_PROCESS_GROUP_RANKS}, \
            TP zero3 process group: {_TP_ZERO3_PROCESS_GROUP_RANKS}, \
            global rank: {dist.get_rank()}")
    return


def _is_layerzero_pg_initialized():
    return _ZERO1_PROCESS_GROUP is not None and _ZERO3_PROCESS_GROUP is not None


def layerzero_initialize_model_parallel_wrapper(initialize_model_parallel):
    @wraps(initialize_model_parallel)
    def wrapper(*args, **kargs):
        results = initialize_model_parallel(*args, **kargs)
        global_args = get_args()
        if getattr(global_args, 'layerzero', False):
            print_rank_0(
                f"Entering initialize_model_parallel to create layerzero process groups")
            config_yaml = global_args.layerzero_config
            config = LayerzeroConfig.load_from_yaml(config_yaml)
            zero3_size = config.zero3_size
            initialized_zero_process_group(zero3_size)
        return results

    return wrapper


def ensure_divisibility(a: int, b: int):
    """Ensure that 'a' is divisible by 'b'. If not, raise an AssertionError with a custom or default message."""
    if b == 0:
        raise ValueError("The divisor (b) must not be zero.")
    if a % b != 0:
        raise ValueError(f"{a} is not divisible by {b}")