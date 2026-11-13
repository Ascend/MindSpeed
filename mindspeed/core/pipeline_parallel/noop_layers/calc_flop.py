# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""Calculate total number of floating point openrations."""

from argparse import Namespace
from typing import Callable


def calc_flop(func: Callable, args: Namespace, batch_size: int) -> float:
    """Calculate total number of floating point operations of the model
        by consider the noop transformer layers situation.

    Args:
        func (Callable): A function to calculate flop
            without considering noop tranformer layer.
            the func have two arguments such as `func(args, batch_size)`.
        args (Namespace): Arguments from cli or configure file.
        batch_size (int): batch size of dataset.

    Returns:
        float: Total number of floating point operations
            considering noop transformer layers.
    """
    is_noop_layers_set = isinstance(args.noop_layers, set)
    args.num_layers -= len(args.noop_layers) if is_noop_layers_set else 0
    ret = func(args, batch_size)
    args.num_layers += len(args.noop_layers) if is_noop_layers_set else 0
    return ret
