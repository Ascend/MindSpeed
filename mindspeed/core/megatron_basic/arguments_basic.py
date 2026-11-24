# Copyright 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
