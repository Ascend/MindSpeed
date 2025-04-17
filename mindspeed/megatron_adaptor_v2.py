"""Adaptor for all megatron functions by feature granularity."""

from argparse import Namespace
from typing import Type
import os
import sys
import shutil
from multiprocessing import Lock
from logging import getLogger
from pathlib import Path

from torch.utils.cpp_extension import _get_build_directory
from torch_npu.contrib import transfer_to_npu

from .deprecate import DisableExecution

# just in case execution of exe_adaptation function
# in megatron_adaptor when import it.

from mindspeed.features_manager import FEATURES_LIST_V2
DisableExecution.DISABLE = True
from mindspeed.patch_utils import MindSpeedPatchesManager
from mindspeed.arguments_v2 import get_mindspeed_args

DisableExecution.DISABLE = False
from mindspeed.log_config import set_log_config

LOG = getLogger(__name__)
_IS_FEATURES_PATCHED = False


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
    for feature in FEATURES_LIST_V2:
        if feature.is_need_apply(mindspeed_args):
            feature.pre_register_patches(MindSpeedPatchesManager, mindspeed_args)
    MindSpeedPatchesManager.apply_patches()

    # apply megatron patches
    for feature in FEATURES_LIST_V2:
        if feature.is_need_apply(mindspeed_args):
            feature.register_patches(MindSpeedPatchesManager, mindspeed_args)
    MindSpeedPatchesManager.apply_patches()

    # accelerate package will check TE on sys.modules，so we need remove this patch
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
