"""Adaptor for all megatron functions by feature granularity."""

import os
import sys
import shutil
from multiprocessing import Lock
from logging import getLogger
from pathlib import Path

from torch.utils.cpp_extension import _get_build_directory
from torch_npu.contrib import transfer_to_npu


from mindspeed.log_config import set_log_config
from mindspeed.deprecate import AutoExecuteFunction, NoExecuteFunction
from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager

with NoExecuteFunction():
    from mindspeed.megatron_adaptor import get_mindspeed_args

LOG = getLogger(__name__)
_IS_FEATURES_PATCHED = False


@AutoExecuteFunction
def patch_features():
    """Patch all mindspeed related features."""
    global _IS_FEATURES_PATCHED
    if _IS_FEATURES_PATCHED:
        return
    _IS_FEATURES_PATCHED = True

    set_log_config()
    log = getLogger(__name__)
    log.info("start to patch features in megatron adaptor v2.")

    mindspeed_args = get_mindspeed_args()
    delete_lock_file()

    # apply patches before import megatron
    MindSpeedFeaturesManager.apply_features_pre_patches(mindspeed_args)

    # apply megatron patches
    MindSpeedFeaturesManager.apply_features_patches(mindspeed_args)

    # accelerate package will check TE on sys.modules, so we need remove this patch
    if 'transformer_engine' in sys.modules:
        del sys.modules["transformer_engine"]


def delete_lock_file():
    """Delete lock file in multiprocess for JIT build.."""
    directory = Path(_get_build_directory("", True))
    if not directory.exists():
        return
    with Lock():
        files = [item for item in directory.iterdir() if item.is_file() and item.name.endswith("lock")]
        if files:
            LOG.info("Process (PID:%s is deleting Lock directory", os.getpid())
            shutil.rmtree(directory)


patch_features()
