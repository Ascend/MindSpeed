#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
_cached_modules_map = {}
_cached_parameters_map = {}
_cached_named_parameters_map = {}


def cached_modules(model):
    key = id(model)
    if key not in _cached_modules_map:
        _cached_modules_map[key] = list(model.modules())
    return _cached_modules_map[key]


def cached_parameters(model):
    key = id(model)
    if key not in _cached_parameters_map:
        _cached_parameters_map[key] = list(model.parameters())
    return _cached_parameters_map[key]


def cached_named_parameters(model):
    key = id(model)
    if key not in _cached_named_parameters_map:
        _cached_named_parameters_map[key] = list(model.named_parameters())
    return _cached_named_parameters_map[key]
