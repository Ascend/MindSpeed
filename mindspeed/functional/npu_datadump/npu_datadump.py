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
        MSTT_DEBUGGER.start()
        fn(*args, **kwargs)
    return wrapper


def dump_end_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn(*args, **kwargs)
        MSTT_DEBUGGER.stop()
        MSTT_DEBUGGER.step()
    return wrapper