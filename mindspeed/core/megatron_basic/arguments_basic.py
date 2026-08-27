# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from logging import getLogger
from functools import wraps

LOG = getLogger(__name__)


def print_after_validate_wrapper(fn):
    """Only print the arguments after the inner validation finishes.

    Feature argument registration and validation are handled by MA's
    ``parse_args_wrapper``/``validate_args_wrapper`` over the unified feature
    list, so the MindSpeed layer keeps only the after-validation printing.
    """

    @wraps(fn)
    def wrapper(args, defaults=None):
        args = fn(args, defaults)

        # _print_args is patched, so it has three arguments.
        from megatron.training.arguments import _print_args

        _print_args("arguments", args, True)

        return args

    return wrapper


def print_args_wrapper(fn):
    @wraps(fn)
    def wrapper(title, args, after_validate=False):
        if after_validate:
            fn(title, args)

    return wrapper
