# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# pylint: skip-file

import copy
import warnings
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from mindspeed.core.optimizer.muon.emerging_optimizers import (
    HAVE_EMERGING_OPTIMIZERS,
    _EMERGING_OPTIMIZERS,
    _create_emerging_optimizer,
)
from mindspeed.core.optimizer.muon.layer_wise_optimizer import LayerWiseDistributedOptimizer
from mindspeed.core.optimizer.muon.optimizer_config import (
    ParamKey,
    ParamPredicate,
    ParamWithNamePredicate,
)
from mindspeed.core.optimizer.muon.optimizer_param_scheduler import (
    ParamGroupOverride,
    combine_param_group_overrides,
    param_group_override_to_tuple,
)
from mindspeed.core.optimizer.muon.utils import LegacyProcessGroupCollection


_MUON_TENSOR_MODEL_PARALLEL_ATTRIBUTES = ("expert_tp", "is_qkv")


def add_muon_tensor_model_parallel_attributes():
    """Patch Megatron 0.12.1 tensor-parallel attribute defaults in memory."""
    from megatron.core.tensor_parallel import layers

    for attribute in _MUON_TENSOR_MODEL_PARALLEL_ATTRIBUTES:
        layers._MODEL_PARALLEL_ATTRIBUTE_DEFAULTS.setdefault(attribute, False)


def copy_muon_tensor_model_parallel_attributes_wrapper(func):
    """Copy Muon tensor-parallel metadata when Megatron creates master params."""

    @wraps(func)
    def wrapper(destination_tensor, source_tensor):
        result = func(destination_tensor, source_tensor)
        for attribute in _MUON_TENSOR_MODEL_PARALLEL_ATTRIBUTES:
            if hasattr(source_tensor, attribute):
                setattr(destination_tensor, attribute, getattr(source_tensor, attribute))
        return result

    return wrapper


def param_is_not_tensor_parallel_duplicate(param, tp_group=None):
    """Dev-style TP duplicate filter with a Megatron 0.12.1 fallback."""
    from megatron.core import mpu

    if hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel:
        return True
    if tp_group is not None:
        return torch.distributed.get_rank(group=tp_group) == 0
    return mpu.get_tensor_model_parallel_rank() == 0


def get_main_grads_for_grad_norm(self) -> List[torch.Tensor]:
    """MegatronOptimizer.get_main_grads_for_grad_norm with explicit tp_group."""
    from megatron.core.transformer.module import param_is_not_shared

    grads_for_norm = []
    for param in self.get_parameters():
        if getattr(self.config, "use_precision_aware_optimizer", False):
            grad = param.decoupled_grad if hasattr(param, "decoupled_grad") else None
        else:
            grad = param.grad
        if (
            grad is not None
            and param_is_not_shared(param)
            and param_is_not_tensor_parallel_duplicate(param, getattr(self, "tp_group", None))
        ):
            grads_for_norm.append(grad)
    return grads_for_norm


def count_zeros_fp32(
    parameters,
    grad_stats_parallel_group: torch.distributed.ProcessGroup,
    use_decoupled_grad: bool = False,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Count zero grads with explicit TP duplicate filtering."""
    from megatron.core.transformer.module import param_is_not_shared
    from megatron.core.utils import get_data_parallel_group_if_dtensor, to_local_if_dtensor

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    total_num_zeros = torch.tensor([0.0], dtype=torch.float, device="cuda")
    data_parallel_group = None
    for param in parameters:
        grad_attr = "decoupled_grad" if use_decoupled_grad else "grad"
        grad_not_none = hasattr(param, grad_attr) and getattr(param, grad_attr) is not None
        if grad_not_none and param_is_not_shared(param) and param_is_not_tensor_parallel_duplicate(param, tp_group):
            grad_obj = getattr(param, grad_attr)
            data_parallel_group = get_data_parallel_group_if_dtensor(grad_obj, data_parallel_group)
            grad = to_local_if_dtensor(grad_obj).detach()
            total_num_zeros += grad.numel() - torch.count_nonzero(grad)

    if data_parallel_group:
        torch.distributed.all_reduce(total_num_zeros, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group)
    torch.distributed.all_reduce(total_num_zeros, op=torch.distributed.ReduceOp.SUM, group=grad_stats_parallel_group)
    return total_num_zeros.item()


def megatron_optimizer_count_zeros(self) -> float:
    """MegatronOptimizer.count_zeros with explicit tp_group."""
    return count_zeros_fp32(
        self.get_parameters(),
        grad_stats_parallel_group=self.get_grad_stats_parallel_group(),
        use_decoupled_grad=getattr(self.config, "use_precision_aware_optimizer", False),
        tp_group=getattr(self, "tp_group", None),
    )


def chained_optimizer_count_zeros(self):
    """Avoid losing per-optimizer tp_group in ChainedOptimizer.count_zeros."""
    num_zeros_in_grad = 0
    for optimizer in self.chained_optimizers:
        num_zeros_in_grad += optimizer.count_zeros() if optimizer.config.log_num_zeros_in_grad else 0
    return num_zeros_in_grad


def _get_muon_config_overrides(
    config,
    no_weight_decay_cond: Optional[Callable],
    scale_lr_cond: Optional[Callable],
    lr_mult: float,
) -> Dict[ParamKey, ParamGroupOverride]:
    config_overrides = {}

    if no_weight_decay_cond is not None:
        no_wd_param = ParamWithNamePredicate(
            name="no_weight_decay_cond",
            fn=lambda param, name: no_weight_decay_cond(name, param),
        )
        param_wd_mult_key = ParamKey(with_name_predicate=no_wd_param)
    elif getattr(config, "apply_wd_to_qk_layernorm", False):
        shape_1_not_qkln_param = ParamWithNamePredicate(
            name="s1_not_qkln",
            fn=lambda param, name: (
                (len(param.shape) == 1 or name.endswith(".bias"))
                and not ("q_layernorm." in name or "k_layernorm." in name)
            ),
        )
        param_wd_mult_key = ParamKey(with_name_predicate=shape_1_not_qkln_param)
    else:
        param_length_1_match = ParamPredicate(name="param_len_1", fn=lambda param: len(param.shape) == 1)
        param_wd_mult_key = ParamKey(name="*.bias", predicate=param_length_1_match)
    config_overrides[param_wd_mult_key] = ParamGroupOverride(wd_mult=0.0)

    if scale_lr_cond is not None:
        scale_lr_param = ParamWithNamePredicate(
            name="scale_lr_cond",
            fn=lambda param, name: scale_lr_cond(name, param),
        )
        config_overrides[ParamKey(with_name_predicate=scale_lr_param)] = ParamGroupOverride(lr_mult=lr_mult)

    if getattr(config, "decoupled_lr", None) is not None:
        decoupled_lr_config = ParamGroupOverride(max_lr=config.decoupled_lr)
        if getattr(config, "decoupled_min_lr", None) is not None:
            decoupled_lr_config["min_lr"] = config.decoupled_min_lr
        config_overrides[ParamKey(attr="is_embedding_or_output_parameter")] = decoupled_lr_config

    return config_overrides


def _get_param_groups(
    model_chunks: List,
    config,
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]],
) -> List[Dict]:
    """Create parameter groups for optimizer.

    Creates parameter groups from provided optimizer config object.

    NOTE There can be more than one match between a ParamKey and a parameter.
        What we do is merge all of the matching ParamKey overrides into a single ParamGroupOverride
        for that parameter and use that as the key for that parameter. Any parameters that get
        the same set of merged overrides will be mapped into the same parameter group.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        config (OptimizerConfig): optimizer configuration object.
        config_overrides (Optional[Dict[ParamKey, ParamGroupOverride]): optimizer overrides,
            specified on a per-layer basis. NOTE: if you want to skip applying weight decay on bias
            and length 1 parameters, and also do not want to do any other overrides, set this to an
            empty dictionary rather than the default value of None.
    Returns:
        List of parameter groups.
    """

    # Map (pg_overrides, is_expert_parallel) to params.
    params_map = {}

    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            # Get optimizer config overrides for this parameter.
            param_overrides_list: List[ParamGroupOverride] = []
            if config_overrides is not None:
                for param_key, param_override in config_overrides.items():
                    if param_key.matches(param, name):
                        param_overrides_list.append(param_override)

            if param_overrides_list:
                param_override: Optional[ParamGroupOverride] = combine_param_group_overrides(param_overrides_list)
            else:
                param_override = None

            is_expert_parallel = not getattr(param, "allreduce", True)

            # Create config_tuple that is hash-able, and has a consistent ordering of the keys.
            param_override_tuple: Optional[Tuple[Tuple[str, Any], ...]] = param_group_override_to_tuple(param_override)
            key = (param_override_tuple, is_expert_parallel)
            if key not in params_map:
                params_map[key] = []
            params_map[key].append(param)

    # Distributed checkpoint requires all ranks to have the same param groups,
    # so we need to align the param groups across ranks, otherwise we may have
    # runtime error when loading the checkpoint or numerical error when resuming training.
    params_key = list(params_map.keys())
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered_params_key = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_params_key, params_key)
        for keys in gathered_params_key:
            for key in keys:
                if key not in params_key:
                    params_key.append(key)
    # Need to pick one of the param_override_tuples to use for the param group.
    param_groups = []
    # Sort keys, None first.
    for key in sorted(params_key, key=lambda x: (x[0] is not None, x[0])):
        param_override_tuple, is_expert_parallel = key
        params = params_map[key] if key in params_map else []
        if param_override_tuple is None:
            param_override = ParamGroupOverride()
        else:
            param_override = ParamGroupOverride({k: v for (k, v) in param_override_tuple})

        # False if param_group_override is None or empty tuple or if we do not modify the
        #  LR schedule.
        #  NOTE: "default_config" is used for logging the learning rate in training.py.
        #   so set to True if we do not modify the learning rate.
        #  if param_group['default_config']:
        #    learning_rate = param_group['lr']
        uses_default_lr_schedule: bool = (not bool(param_override_tuple)) or not any(
            ["lr" in k for k in param_override]
        )

        # TODO: Remove "backwards compatible" fields below eventually.
        default_config = ParamGroupOverride(
            wd_mult=1.0,
            lr_mult=1.0,
            is_decoupled_lr=False,
            # The following two fields may be important to keep even when we remove the
            #   above "backwards compatible" fields.
            max_lr=config.lr,  # user may override this in param_override
            min_lr=config.min_lr,  # user may override this in param_override
        )
        if "params" in param_override:
            raise ValueError("'params' should not be in param_override, this is a protected key")
        param_group = {
            "params": params,
            "is_expert_parallel": is_expert_parallel,
            "default_config": uses_default_lr_schedule,
            **default_config,
            **param_override,  # keep **param_override last so that users can override other fields.
        }
        param_groups.append(param_group)

    return param_groups


def get_megatron_optimizer_based_on_param_groups_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import inspect

        signature = inspect.signature(func)
        if "skip_megatron_wrapping" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        ):
            return func(*args, **kwargs)

        skip_megatron_wrapping = kwargs.pop("skip_megatron_wrapping", False)
        kwargs.pop("pg_collection", None)
        if not skip_megatron_wrapping:
            return func(*args, **kwargs)

        config = kwargs["config"] if "config" in kwargs else args[0]
        param_groups = kwargs["param_groups"] if "param_groups" in kwargs else args[2]

        if getattr(config, "use_precision_aware_optimizer", False):
            raise ValueError("skip_megatron_wrapping=True is incompatible with use_precision_aware_optimizer.")
        if getattr(config, "optimizer_cpu_offload", False):
            raise ValueError("skip_megatron_wrapping=True is incompatible with optimizer_cpu_offload.")

        import megatron.core.optimizer as optimizer_mod

        if param_groups:
            if config.optimizer == 'adam':
                kwargs = {
                    "params": param_groups,
                    "lr": config.lr,
                    "weight_decay": config.weight_decay,
                    "betas": (config.adam_beta1, config.adam_beta2),
                    "eps": config.adam_eps,
                }
                if hasattr(config, "optimizer_cuda_graph"):
                    kwargs["capturable"] = config.optimizer_cuda_graph

                adam_cls = optimizer_mod.Adam
                try:
                    supports_adam_w_mode = "adam_w_mode" in inspect.signature(adam_cls.__init__).parameters
                except (TypeError, ValueError):
                    supports_adam_w_mode = not adam_cls.__module__.startswith("torch.optim")

                if supports_adam_w_mode:
                    kwargs["adam_w_mode"] = getattr(config, "decoupled_weight_decay", True)
                elif adam_cls.__module__.startswith("torch.optim"):
                    adam_cls = (
                        torch.optim.AdamW if getattr(config, "decoupled_weight_decay", True) else torch.optim.Adam
                    )
                elif not getattr(config, "decoupled_weight_decay", True):
                    adam_cls = torch.optim.Adam

                optimizer = adam_cls(**kwargs)

                def init_state_fn(opt, config=None):
                    for group in opt.param_groups:
                        for p in group['params']:
                            if len(opt.state[p]) == 0:
                                if config is None or not config.use_precision_aware_optimizer:
                                    opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                                    opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                                else:
                                    opt.initialize_state(p)

            elif config.optimizer == 'lion':
                try:
                    from emerging_optimizers.scalar_optimizers import Lion
                except ImportError as exc:
                    raise ImportError(
                        "Lion optimizer requires emerging_optimizers >= 0.2. "
                        "Please install or upgrade it to use --optimizer lion."
                    ) from exc
                optimizer = Lion(
                    param_groups,
                    lr=config.lr,
                    betas=(
                        getattr(config, "lion_beta1", 0.95),
                        getattr(config, "lion_beta2", 0.98),
                    ),
                    weight_decay=config.weight_decay,
                )

                def init_state_fn(opt, config=None):
                    for group in opt.param_groups:
                        for p in group['params']:
                            if len(opt.state[p]) == 0:
                                opt.state[p]['exp_avg'] = torch.zeros_like(p.data)

            elif config.optimizer == 'sgd':
                optimizer = optimizer_mod.SGD(
                    param_groups,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    momentum=config.sgd_momentum,
                )
                init_state_fn = None
            else:
                raise Exception('{} optimizer is not supported.'.format(config.optimizer))
        else:
            optimizer = None
            init_state_fn = None

        return optimizer, init_state_fn

    return wrapper


def _get_megatron_emerging_optimizer(
    config,
    model_chunks: List,
    config_overrides: Optional[Dict[ParamKey, Any]] = None,
    pg_collection: Optional[LegacyProcessGroupCollection] = None,
):
    """Build an emerging optimizer using Megatron dev's high-level flow."""
    from megatron.core.optimizer import _get_megatron_optimizer_based_on_param_groups
    from megatron.core.optimizer.optimizer import (
        ChainedOptimizer,
        Float16OptimizerWithFloat16Params,
        FP32Optimizer,
    )

    eopt_name = config.optimizer
    use_layer_wise = bool(getattr(config, "use_layer_wise_distributed_optimizer", False))
    if eopt_name.startswith("dist_"):
        bare_name = eopt_name[len("dist_") :]
        warnings.warn(
            f"optimizer='{eopt_name}' is deprecated. Use optimizer='{bare_name}' "
            "with use_layer_wise_distributed_optimizer=True.",
            DeprecationWarning,
            stacklevel=3,
        )
        eopt_name = bare_name
        use_layer_wise = True
    if not HAVE_EMERGING_OPTIMIZERS:
        raise ImportError(f"MindSpeed local emerging optimizer implementation is required for optimizer='{eopt_name}'.")
    if eopt_name not in _EMERGING_OPTIMIZERS:
        raise ValueError(f"Unsupported emerging optimizer: {eopt_name}")
    if getattr(config, "fp16", False):
        raise ValueError("emerging optimizer with fp16 is not supported.")

    if pg_collection is None:
        pg_collection = LegacyProcessGroupCollection()

    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if "experts" in name and "shared" not in name:
                param.expert_tp = True
            # TODO(deyuf): support MLA
            if "linear_qkv.weight" in name and len(param.shape) == 2:
                param.is_qkv = True

    if config_overrides is None:
        config_overrides = {}
    config_overrides.update(_EMERGING_OPTIMIZERS[eopt_name].default_param_overrides)

    all_param_groups = _get_param_groups(model_chunks, config, config_overrides)
    grouped_param_groups = defaultdict(list)
    for group in all_param_groups:
        opt_name = group.get("optimizer", eopt_name)
        is_expert = group["is_expert_parallel"] and not use_layer_wise
        grouped_param_groups[(opt_name, is_expert)].append(group)

    results = []
    for (opt_name, is_expert), groups in grouped_param_groups.items():
        if not groups:
            continue

        model_parallel_group = pg_collection.tp_ep_pp if is_expert else pg_collection.mp
        if opt_name in _EMERGING_OPTIMIZERS:
            optimizer, init_state_fn = _create_emerging_optimizer(
                config, groups, eopt_name, model_chunks, pg_collection
            )
            if use_layer_wise:
                result = (optimizer, init_state_fn)
            else:
                if getattr(config, "bf16", False):
                    optimizer = Float16OptimizerWithFloat16Params(optimizer, config, None, init_state_fn)
                else:
                    optimizer = FP32Optimizer(optimizer, config, init_state_fn)
                setattr(optimizer, "grad_stats_parallel_group", model_parallel_group)
                setattr(optimizer, "tp_group", pg_collection.tp)
                result = optimizer
        else:
            fallback_config = copy.copy(config)
            fallback_config.optimizer = opt_name
            fallback_config.use_distributed_optimizer = False
            result = _get_megatron_optimizer_based_on_param_groups(
                config=fallback_config,
                model_chunks=model_chunks,
                param_groups=groups,
                model_parallel_group=model_parallel_group,
                pg_collection=pg_collection,
                skip_megatron_wrapping=use_layer_wise,
            )
            if use_layer_wise and not isinstance(result, tuple):
                raise RuntimeError(
                    "Megatron _get_megatron_optimizer_based_on_param_groups must "
                    "support skip_megatron_wrapping for Muon layer-wise scalar fallback."
                )
            if not use_layer_wise and hasattr(result, "config"):
                result.config = config
        results.append(result)

    if use_layer_wise:
        base_optimizers, init_fns = (), ()
        if results:
            base_optimizers, init_fns = zip(*results)
        return LayerWiseDistributedOptimizer(
            list(base_optimizers),
            config,
            pg_collection=pg_collection,
            init_state_fn_list=list(init_fns),
            model_chunks=model_chunks if getattr(config, "overlap_param_gather", False) else None,
        )

    return ChainedOptimizer(results)


def get_megatron_optimizer_muon_wrapper(func):
    """Intercept Megatron's optimizer factory when --optimizer muon is selected."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args:
            config = args[0]
            model_chunks = args[1] if len(args) > 1 else kwargs.get("model_chunks")
        else:
            config = kwargs.get("config")
            model_chunks = kwargs.get("model_chunks")

        optimizer_name = getattr(config, "optimizer", None)
        if optimizer_name not in ("muon", "dist_muon"):
            return func(*args, **kwargs)

        no_weight_decay_cond = kwargs.get("no_weight_decay_cond")
        scale_lr_cond = kwargs.get("scale_lr_cond")
        lr_mult = kwargs.get("lr_mult", 1.0)
        if len(args) > 2:
            no_weight_decay_cond = args[2]
        if len(args) > 3:
            scale_lr_cond = args[3]
        if len(args) > 4:
            lr_mult = args[4]

        config_overrides = _get_muon_config_overrides(config, no_weight_decay_cond, scale_lr_cond, lr_mult)

        return _get_megatron_emerging_optimizer(
            config=config,
            model_chunks=model_chunks,
            config_overrides=config_overrides,
        )

    return wrapper
