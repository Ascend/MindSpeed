# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import functools
import inspect
import sys
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


def _unwrap_callable(func):
    seen = set()
    while isinstance(func, functools.partial):
        if id(func) in seen:
            return func
        seen.add(id(func))
        func = func.func
    while hasattr(func, "__wrapped__"):
        if id(func) in seen:
            return func
        seen.add(id(func))
        func = func.__wrapped__
        while isinstance(func, functools.partial):
            if id(func) in seen:
                return func
            seen.add(id(func))
            func = func.func
    return func


def _iter_related_callables(func, seen=None):
    seen = seen or set()
    func = _unwrap_callable(func)
    func_id = id(func)
    if func_id in seen:
        return
    seen.add(func_id)
    yield func

    closure = getattr(func, "__closure__", None)
    if closure is not None:
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if callable(value):
                yield from _iter_related_callables(value, seen)

    for attr_name in ("forward_step_func", "func", "_func"):
        value = getattr(func, attr_name, None)
        if callable(value):
            yield from _iter_related_callables(value, seen)


def _get_forward_step_func(pretrain, args, kwargs):
    if "forward_step_func" in kwargs:
        return kwargs["forward_step_func"]

    try:
        bound_args = inspect.signature(pretrain).bind_partial(*args, **kwargs).arguments
    except (TypeError, ValueError):
        bound_args = {}
    if "forward_step_func" in bound_args:
        return bound_args["forward_step_func"]

    # Megatron-LM 0.17/0.18 keep forward_step_func close to the model_type argument.
    for index in (4, 3):
        if len(args) > index and callable(args[index]):
            return args[index]
    return None


def _find_entrypoint_globals(forward_step_func):
    for func in _iter_related_callables(forward_step_func):
        module_name = getattr(func, "__module__", None)
        module = sys.modules.get(module_name)
        module_globals = getattr(func, "__globals__", None)
        if module_globals is None and module is not None:
            module_globals = getattr(module, "__dict__", None)
        if module_globals is None:
            continue
        if "get_batch" in module_globals and "core_transformer_config_from_args" in module_globals:
            return module_globals

    main_globals = getattr(sys.modules.get("__main__"), "__dict__", None)
    if main_globals is not None:
        if "get_batch" in main_globals and "core_transformer_config_from_args" in main_globals:
            return main_globals

    return None


def make_get_batch_config_cache_wrapper(original_factory, get_batch_func):
    if getattr(original_factory, "_mindspeed_get_batch_config_cached", False):
        return original_factory

    get_batch_code = getattr(get_batch_func, "__code__", None)
    cached_configs = {}

    @functools.wraps(original_factory)
    def cached_factory(args, config_class=None):
        frame = inspect.currentframe()
        try:
            caller_code = (
                getattr(frame.f_back, "f_code", None) if frame is not None and frame.f_back is not None else None
            )
        finally:
            del frame
        if caller_code is not get_batch_code:
            return original_factory(args, config_class)

        cache_key = (
            id(args),
            config_class,
            getattr(args, "multi_latent_attention", None),
            getattr(args, "heterogeneous_layers_config_path", None),
        )
        if cache_key not in cached_configs:
            cached_configs[cache_key] = original_factory(args, config_class)
        return cached_configs[cache_key]

    cached_factory._mindspeed_get_batch_config_cached = True
    cached_factory._mindspeed_get_batch_config_cache = cached_configs
    cached_factory._mindspeed_get_batch_config_original = original_factory
    return cached_factory


def inject_get_batch_config_cache_wrapper(pretrain):
    @functools.wraps(pretrain)
    def wrapped(*args, **kwargs):
        forward_step_func = _get_forward_step_func(pretrain, args, kwargs)
        entrypoint_globals = _find_entrypoint_globals(forward_step_func)
        if entrypoint_globals is not None:
            get_batch_func = entrypoint_globals.get("get_batch")
            original_factory = entrypoint_globals.get("core_transformer_config_from_args")
            if callable(get_batch_func) and callable(original_factory):
                entrypoint_globals["core_transformer_config_from_args"] = make_get_batch_config_cache_wrapper(
                    original_factory, get_batch_func
                )
        return pretrain(*args, **kwargs)

    return wrapped


class CacheGetBatchConfigFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__("cache-get-batch-config", optimization_level=2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            "--cache-get-batch-config",
            action="store_true",
            default=False,
            help="Cache the TransformerConfig constructed from get_batch to reduce host overhead.",
        )

    def register_patches(self, patch_manager, args):
        if getattr(args, self.feature_name, None):
            patch_manager.register_patch(
                "megatron.training.training.pretrain",
                inject_get_batch_config_cache_wrapper,
            )
