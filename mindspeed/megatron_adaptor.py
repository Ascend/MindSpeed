"""Adaptor for all megatron functions by feature granularity."""

import os
import sys
import shutil
from multiprocessing import Lock
from logging import getLogger
from pathlib import Path

from torch.utils.cpp_extension import _get_build_directory
from torch_npu.contrib import transfer_to_npu

from mindspeed.args_utils import get_full_args
from mindspeed.args_utils import get_mindspeed_args
from mindspeed.log_config import set_log_config
from mindspeed.deprecate import AutoExecuteFunction
from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager

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
    log.info("start to patch features in megatron adaptor.")

    mindspeed_args = get_mindspeed_args()
    delete_lock_file()

    # apply patches before import megatron
    MindSpeedFeaturesManager.apply_features_pre_patches(mindspeed_args)

    # apply megatron patches
    MindSpeedFeaturesManager.apply_features_patches(mindspeed_args)


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


def repatch(args):
    MindSpeedFeaturesManager.remove_patches()
    full_args = get_full_args()
    for k, v in args.items():
        setattr(full_args, k, v)
    MindSpeedFeaturesManager.apply_features_pre_patches(full_args)
    MindSpeedFeaturesManager.apply_features_patches(full_args)


patch_features()
