#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import zlib
import torch
from mindspore import Tensor


def get_md5(x):
    """calculate md5 value of x, this function can be used both in mindspore and pytorch"""
    if x is None:
        return None
    if not isinstance(x, Tensor):
        raise ValueError(f'the input of get_md5 function should be Tensor, but it\'s type is {type(x)}')
    # detach() and cpu() can be ignored in mindspore, but it's needed in pytorch
    x_np = x.to(torch.float32).detach().cpu().numpy()
    md5 = zlib.crc32(x_np.tobytes())
    return md5


def print_by_rank(content, select_ranks=None, flush=True):
    """print log of workers you choose"""
    if select_ranks is None:
        return
    local_rank = torch.distributed.get_rank()
    if local_rank in select_ranks:
        # local_rank can be ignored in mindspore, but it's needed in pytorch to separate different worker's log
        print(f'rank:{local_rank}, {content}', flush=flush)


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
