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
from functools import wraps
from torch_npu.utils.flops_count import FlopsCounter


# Global FLOPS counter instance
FLOPS_COUNTER = None
RECORDED_COUNT = 0
TRAVERSED_COUNT = 0


def get_count():
    """Get recorded and traversed FLOPS counts."""
    if RECORDED_COUNT == 0 and TRAVERSED_COUNT == 0:
        flops_counter = get_flops_counter()
        count = flops_counter.get_flops()
        set_count(count)
    return RECORDED_COUNT, TRAVERSED_COUNT


def set_count(count):
    """Set the FLOPS counts."""
    global RECORDED_COUNT
    global TRAVERSED_COUNT
    RECORDED_COUNT = count[0]
    TRAVERSED_COUNT = count[1]


def get_flops_counter():
    """Get or initialize the FLOPS counter."""
    global FLOPS_COUNTER
    if FLOPS_COUNTER is None:
        FLOPS_COUNTER = FlopsCounter()
    return FLOPS_COUNTER


def checkpoint_function_backward_wrapper(fn):
    @wraps(fn)
    def wrapper(ctx, *args):
        flops_counter = get_flops_counter()
        flops_counter.pause()
        result = fn(ctx, *args)
        flops_counter.resume()
        return result

    return wrapper


def train_step_wrapper(train_step):
    @wraps(train_step)
    def wrapper(*args, **kwargs):
        flop_count = get_flops_counter()
        flop_count.start()
        ret = train_step(*args, **kwargs)
        counts = flop_count.get_flops()
        set_count(counts)
        flop_count.stop()
        return ret

    return wrapper
