# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

import warnings
from typing import Dict, List, Optional, Iterable, Union, Any

import torch
import torch.nn as nn

from ..zero3._common_utils import (
    clean_tensor_name,
    _named_parameters_with_duplicates
)


@torch.no_grad()
def _shard_optim_state_dict(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    optim_state_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """

    Args:
        model (nn.Module): Root module (which may or may not be a
            :class:`FullyShardedDataParallel` instance) whose parameters
            were passed into the optimizer ``optim``.
        optim (torch.optim.Optimizer): Optimizer for ``model`` 's
            parameters.
        rank0_only (bool): If ``True``, saves the populated :class:`dict`
            only on rank 0; if ``False``, saves it on all ranks. (Default:
            ``True``)
        shard_state (bool): If ``True``, shard and distribute all
            non-zero-dimension states.

    Returns:
        Dict[str, Any]: A :class:`dict` containing the optimizer state that is sharded: FQN - > state_dict.
    """
    param_to_fqns = _get_param_to_fqns(model)
    is_named_optimizer = _is_named_optimizer(optim_state_dict)

    param_key_to_param = _get_param_key_to_param(
        optim, model, is_named_optimizer, param_to_fqns
    )
    param_key_to_fqns, missing_keys = _get_param_key_to_fqns(
        param_to_fqns, param_key_to_param)
    if missing_keys:
        warnings.warn(
            f"Missing keys that do not have FQN mappings {missing_keys}")
    return param_key_to_fqns


def _get_param_key_to_fqns(param_to_fqns, param_key_to_param):
    param_key_to_fqns = {}
    missing_keys = set()
    for param_key, param in param_key_to_param.items():
        if param in param_to_fqns:
            param_key_to_fqns[param_key] = param_to_fqns[param]
        else:
            missing_keys.add(param_key)
    return param_key_to_fqns, missing_keys


def _get_param_to_fqns(
    model: torch.nn.Module,
    dedup_shared_params: bool = True,
) -> Dict[nn.Parameter, List[str]]:
    """
    Constructs a mapping from parameter to a list of its \"canonical\" FQNs. Here,
    we use canonical to mean the fully-qualified name assigned to the parameter
    based on its position in the original nn.Module hierarchy before any wrapper
    or parallelism has been applied to it. This is in contrast to FQNs that may be
    generated after parallelisms or wrappers have been applied to the model.

    Each normal parameter maps to a singleton list containing its FQN, while each
    ``FlatParameter`` maps to a list of its original parameter FQNs, which may
    have length greater than one.  All FQNs are prefixed starting from ``model``.
    """
    param_to_fqns = {}
    for param_name, param in _named_parameters_with_duplicates(
        model
    ):
        local_fqns = [param_name]
        global_fqns = [
            clean_tensor_name(name) for name in local_fqns
        ]  # prefixed from the top level `model` (i.e. including `prefix`)
        is_shared_param = param in param_to_fqns
        if not is_shared_param:
            param_to_fqns[param] = global_fqns
        elif not dedup_shared_params:
            param_to_fqns[param].extend(global_fqns)

    return param_to_fqns


def _is_named_optimizer(optim_state_dict: Dict[str, Any]) -> bool:
    """
    Returns whether the state_dict is from a NamedOptimizer.
    This function checks that the keys in the state_dict['state'] are strings
    (which usually are FQNs) versus integers (which usually refer to param_ids
    from a vanilla torch.optim.Optimizer).
    """
    state = optim_state_dict.get("state", None)
    if not state:
        # If we cannot find a state, assume it is not NamedOptimizer as
        # NamedOptimizer has eager initialization.
        return False
    try:
        key = next(iter(state.keys()))
    except Exception as e:
        raise Exception(optim_state_dict) from e  # noqa: TRY002
    return isinstance(key, str)


def _get_param_key_to_param(
    optim: torch.optim.Optimizer,
    model: Optional[nn.Module] = None,
    is_named_optimizer: bool = False,
    param_to_fqns: Optional[Dict[nn.Parameter, List[str]]] = None,
) -> Dict[Union[int, str], nn.Parameter]:
    """
    Constructs a mapping from parameter keys to parameters. For the regular
    optimizers, the keys are parameter IDs. For NamedOptimizer, the keys
    are FQNs. This API may be used both for models with ``FlatParameter`` s and
    without.
    """
    clean_fqn_to_fsdp_fqn: Dict[str, str] = {}
    if is_named_optimizer:
        if param_to_fqns is None or model is None:
            raise AssertionError("The optimizer is a NamedOptimizer, `param_to_fqns` must not be None.")
        for key, _ in _named_parameters_with_duplicates(model):
            clean_fqn_to_fsdp_fqn[clean_tensor_name(key)] = key

    param_key_to_param: Dict[Union[str, int], nn.Parameter] = {}
    pid = 0
    for param_group in optim.param_groups:
        if is_named_optimizer:
            for param in param_group["params"]:
                # use_orig_params case
                if len(param_to_fqns[param]) != 1:
                    raise AssertionError("More than one fqn matches this param")
                key = param_to_fqns[param][0]
                try:
                    key = clean_fqn_to_fsdp_fqn[key]
                except KeyError as e:
                    raise KeyError(
                        f"Can't find {key} from {list(clean_fqn_to_fsdp_fqn.keys())}."
                    ) from e
                param_key_to_param[key] = param
        else:
            for param in param_group["params"]:
                param_key_to_param[pid] = param
                pid += 1

    return param_key_to_param
