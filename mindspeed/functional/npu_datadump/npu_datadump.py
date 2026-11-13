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

import os
from pathlib import Path
from functools import wraps
from logging import getLogger

import torch
import torch_npu
from msprobe.pytorch import PrecisionDebugger


current_dir = Path(__file__).parent.resolve()
MSTT_DEBUGGER = PrecisionDebugger(config_path=str(current_dir) + '/config.json')
LOG = getLogger(__name__)


def dump_start_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        MSTT_DEBUGGER.start(model=args[2])
        return fn(*args, **kwargs)
    return wrapper


def dump_end_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn(*args, **kwargs)
        MSTT_DEBUGGER.stop()
        MSTT_DEBUGGER.step()
    return wrapper