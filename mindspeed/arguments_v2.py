"""Handle cli arguments by features granularity.

Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""

from argparse import ArgumentParser, Namespace
from typing import List
from functools import wraps

from mindspeed.features_manager import FEATURES_LIST_V2
from .arguments import process_args

_ARGS = None


def extra_args_provider_decorator(extra_args_provider):
    """Make a extra args parser  for magatron."""
    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        parser = process_args(parser)
        for feature in FEATURES_LIST_V2:
            feature.register_args(parser)
        return parser

    return wrapper


def parse_args_wrapper(parse_args):
    """Decorate parse_args function of megatron."""

    @wraps(parse_args)
    def wrapper(extra_args_provider=None, ignore_unknown_args=False):
        decorated_provider = extra_args_provider_decorator(extra_args_provider)
        return parse_args(decorated_provider, ignore_unknown_args)

    return wrapper


def core_transformer_config_from_args_wrapper(fn):
    """A decorator for transformer config args."""
    @wraps(fn)
    def wrapper(args):
        config = fn(args)
        config.context_parallel_algo = args.context_parallel_algo
        config.batch_p2p_comm = False
        if args.use_multiparameter_pipeline_model_parallel:
            config.deallocate_pipeline_outputs = False
        return config

    return wrapper


def get_mindspeed_args() -> Namespace:
    """Get cli arguments of mindspeed."""
    global _ARGS

    if not _ARGS:
        parser = ArgumentParser(
            description="MindSpeed Arguments",
            allow_abbrev=False,
        )
        parser = process_args(parser)
        for feature in FEATURES_LIST_V2:
            feature.register_args(parser)
        _ARGS, unknown = parser.parse_known_args()
        parse_unknown_args(_ARGS, unknown)

    return _ARGS


def add_args(args, key, value):
    """Add args to parser."""
    if key is not None:
        key = key[2:].replace("-", "_")
        if value is None:
            value = True
        elif len(value) == 1:
            value = value[0]
        setattr(args, key, value)


def parse_unknown_args(args: Namespace, unknown: List[str]):
    """Parse special unknown args.

    Args:
        args (Namespace): regular arguments.
        unknown (List[str]): special arguments string.
    """
    i = 0
    key, value = None, None
    while i < len(unknown):
        if unknown[i].startswith("--"):
            add_args(args, key, value)
            key = unknown[i]
            value = None
        else:
            if value is None:
                value = [unknown[i]]
            else:
                value.append(unknown[i])
        i += 1
    add_args(args, key, value)


