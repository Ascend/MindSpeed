# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.distributed as dist
from megatron.training.utils import print_rank_0

from .fqn import ShardFlattenInfo
from ..zero3.fsdp import LayerZeRO3
from ..zero3._common_utils import (
    clean_tensor_name,
    _apply_to_modules,
)
from ..zero3._init_utils import _get_ignored_params
from ..runtime._initialize import _lazy_init

TP_SHARD_ARGS = "tensor_model_parallel"


def clean_state_dict(state_dict: Dict):
    sd = OrderedDict()
    for key, param in state_dict.items():
        fqn = clean_tensor_name(key)
        sd[fqn] = param
    return sd


def use_zero1_params(zero3_model: LayerZeRO3):
    if zero3_model._is_root is None:
        _lazy_init(zero3_model, zero3_model)
    for handle in zero3_model._all_handles:
        if handle:
            already_resharded = handle.flat_param.data_ptr(
            ) == handle.flat_param._zero1_shard.data_ptr()
            if already_resharded:
                handle._use_sharded_views()
                return
            else:
                with zero3_model._device_handle.stream(zero3_model._default_stream):
                    event = zero3_model._device_handle.Event()
                    event.record()
                    event.wait()
                handle.reshard(True)
                handle._prefetched = False
                handle._use_sharded_views()
                return


def clean_ignored_modules(zero3_model: LayerZeRO3, state_dict):
    if zero3_model._is_root is None:
        _lazy_init(zero3_model, zero3_model)
    ignored_params = _get_ignored_params(
        zero3_model, zero3_model._ignored_modules, zero3_model._ignored_params)
    ignored_keys = set()
    for key, param in zero3_model.named_parameters():
        if param in ignored_params:
            ignored_keys.add(key)
    new_state_dict = OrderedDict()
    ignored_param_keys = set()
    for key, param in state_dict.items():
        if key in ignored_keys:
            ignored_param_keys.add(key)
        else:
            new_state_dict[key] = param
    print_rank_0(f"Ignored parameter keys: {ignored_param_keys}")
    return new_state_dict


def shard_state_dict(zero3_model: LayerZeRO3, state_dict):
    '''This function returns a dict of FQN to shard info mappings for later converting to megatron ckpt.
    missing keys maybe params that are not managed by Layerzero3, 
    These params later will directly convert to megatron with no-op
    '''
    if zero3_model._is_root is None:
        _lazy_init(zero3_model, zero3_model)
    if not zero3_model._is_root:
        raise ValueError("Expected a root zero3 model")
    shard_infos = _get_param_fqns_to_shards(zero3_model)
    missing_keys = set()
    for key in state_dict.keys():
        fqn = clean_tensor_name(key)
        if fqn not in shard_infos:
            missing_keys.add(fqn)
    print_rank_0(f"Layerzero3 Shard info {missing_keys=}")
    return shard_infos


def _get_param_fqns_to_shards(
    model: torch.nn.Module,
) -> Dict[str, ShardFlattenInfo]:

    def module_fn(module, prefix, tree_level, shard_infos):
        if isinstance(module, LayerZeRO3):
            handle = module._handle
            if handle:
                flat_param = handle.flat_param
                for param, shard_param_info, fqn, shape in zip(
                    flat_param._params,
                    flat_param._shard_param_infos,
                    flat_param._fqns,
                    flat_param._shapes
                ):
                    if hasattr(param, TP_SHARD_ARGS):
                        tensor_model_parallel = param.tensor_model_parallel
                        partition_dim = param.partition_dim
                        partition_stride = param.partition_stride
                    else:
                        tensor_model_parallel = False
                        partition_dim = -1,
                        partition_stride = 1,
                    global_fqn = prefix + fqn
                    shard_infos[global_fqn] = ShardFlattenInfo(
                        shard_param_info.in_shard,
                        shard_param_info.numel_in_shard,
                        shard_param_info.intra_param_start_idx,
                        shard_param_info.intra_param_end_idx,
                        shape,
                        tensor_model_parallel,
                        partition_dim,
                        partition_stride)

    def return_fn(shard_infos):
        return shard_infos

    param_to_unflat_param_names: Dict[torch.nn.Parameter, List[str]] = {}
    return _apply_to_modules(
        model,
        module_fn,
        return_fn,
        [],
        param_to_unflat_param_names,
    )
