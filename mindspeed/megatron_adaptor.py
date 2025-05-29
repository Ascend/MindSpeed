"""Adaptor for all megatron functions by feature granularity."""

import os
import sys
import shutil
import argparse
from multiprocessing import Lock
from logging import getLogger
from pathlib import Path

from torch.utils.cpp_extension import _get_build_directory
from torch_npu.contrib import transfer_to_npu

from mindspeed.log_config import set_log_config
from mindspeed.deprecate import AutoExecuteFunction
from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager
from mindspeed.arguments import process_args

_ARGS = None
LOG = getLogger(__name__)
_IS_FEATURES_PATCHED = False


def add_args(args, key, value):
    if key is not None:
        key = key[2:].replace('-', '_')
        if value is None:
            value = True
        elif len(value) == 1:
            value = value[0]
        setattr(args, key, value)


def parser_unknown_args(args, unknown):
    i = 0
    key = value = None
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


def get_mindspeed_args():
    global _ARGS
    if _ARGS is None:
        parser = argparse.ArgumentParser(description='MindSpeed Arguments', allow_abbrev=False)
        _ARGS, unknown = process_args(parser).parse_known_args()
        parser_unknown_args(_ARGS, unknown)
    return _ARGS


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
