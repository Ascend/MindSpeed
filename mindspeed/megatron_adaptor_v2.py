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

from .deprecate import AdaptorV2

# just in case execution of exe_adaptation function
# in megatron_adaptor when import it.
AdaptorV2.VERSION = "V2"
from mindspeed.features_manager import FEATURES
from mindspeed.patch_utils import MindSpeedPatchesManager
from mindspeed.arguments_v2 import get_mindspeed_args
from mindspeed.megatron_adaptor import (
    te_adaptation,
    apex_adaptation,
    torch_adaptation,
    mcore_models_adaptation_l0,
    mcore_tensor_parallel_adaptation_l0,
    mcore_transformer_adaptation_l0,
    mcore_moe_adaptation_l0,
    legacy_model_transformer_l0,
    mcore_parallel_state_adaptation,
    communication_adaptation,
    coalescing_manager_adaptation,
    optimizer_selection,
)
AdaptorV2.VERSION = "V1"
from mindspeed.log_config import set_log_config

LOG = getLogger(__name__)


def patch_features():
    """Patch all mindspeed related features."""
    set_log_config()
    log = getLogger(__name__)
    log.info("start to patch features in megatron adaptor v2.")

    mindspeed_args = get_mindspeed_args()
    delete_lock_file()

    optimizer_selection(MindSpeedPatchesManager, mindspeed_args)
    adaptation_l0(mspm=MindSpeedPatchesManager, mindspeed_args=mindspeed_args)

    for feature in FEATURES:
        if (
            getattr(mindspeed_args, feature.feature_name, None)
            or feature.default_patches
        ):
            feature.register_patches(MindSpeedPatchesManager, mindspeed_args)

    MindSpeedPatchesManager.apply_patches()
    del sys.modules["transformer_engine"]


def delete_lock_file():
    """Delete lock file in multiprocess for JIT build.."""
    directory = Path(_get_build_directory("", True))
    if not directory.exists():
        return
    with Lock():
        files = [
            item
            for item in directory.iterdir()
            if item.is_file() and item.name.endswith("lock")
        ]
        if files:
            LOG.info("Process (PID:%s is deleting Lock directory", os.getpid())
            shutil.rmtree(directory)


def adaptation_l0(
    mspm: Type[MindSpeedPatchesManager],
    mindspeed_args: Namespace,
):
    """The minimum patch set for megatron to adapt to NPU."""
    # transformer_engine
    te_adaptation(mspm)
    apex_adaptation(mspm)
    torch_adaptation(mspm)
    # Need replace transformer_engine modules before import megatron
    mspm.apply_patches()

    mcore_models_adaptation_l0(mspm)
    mcore_tensor_parallel_adaptation_l0(mspm)
    mcore_transformer_adaptation_l0(mspm)
    mcore_moe_adaptation_l0(mspm)
    legacy_model_transformer_l0(mspm)
    megatron_training_adaptation_l0(mspm)
    # context parallel(ring attention) requires mcore parallel state patch
    mcore_parallel_state_adaptation(mspm)
    # just make communication_adaptation work
    mindspeed_args.disable_gloo_group = None
    communication_adaptation(mspm, mindspeed_args)
    coalescing_manager_adaptation(mspm, mindspeed_args)


def megatron_training_adaptation_l0(mspm: Type[MindSpeedPatchesManager]):
    """Implement training adaption l0."""
    from .initialize import (
        _compile_dependencies,
        set_jit_fusion_options_wrapper,
    )
    from .utils import get_batch_on_this_cp_rank
    from .training import pretrain, get_device_wrapper
    from .arguments_v2 import (
        parse_args_wrapper,
        core_transformer_config_from_args_wrapper,
        validate_args_wrapper,
    )
    from .yaml_arguments import (
        core_transformer_config_from_yaml_wrapper,
        print_args_wrapper,
    )

    from .core.training import train_decorator, train_step_decorator
    from .core.transformer.transformer_config import (
        transformer_config_post_init_wrapper,
    )

    mspm.register_patch(
        "megatron.training.training.train",
        train_decorator,
    )
    mspm.register_patch(
        "megatron.training.training.train_step",
        train_step_decorator,
    )
    mspm.register_patch(
        "megatron.training.yaml_arguments.core_transformer_config_from_yaml",
        core_transformer_config_from_yaml_wrapper,
    )
    mspm.register_patch(
        "megatron.training.initialize._compile_dependencies",
        _compile_dependencies,
    )
    mspm.register_patch(
        "megatron.training.utils.get_batch_on_this_cp_rank",
        get_batch_on_this_cp_rank,
    )
    mspm.register_patch(
        "megatron.training.arguments.parse_args",
        parse_args_wrapper,
    )
    mspm.register_patch(
        "megatron.training.arguments.validate_args", validate_args_wrapper
    )
    mspm.register_patch(
        "megatron.training.arguments._print_args",
        print_args_wrapper,
    )
    mspm.register_patch(
        "megatron.training.yaml_arguments.validate_yaml", validate_args_wrapper
    )
    mspm.register_patch(
        "megatron.training.yaml_arguments._print_args", print_args_wrapper
    )
    mspm.register_patch(
        "megatron.training.arguments.core_transformer_config_from_args",
        core_transformer_config_from_args_wrapper,
    )
    mspm.register_patch(
        "megatron.training.initialize.set_jit_fusion_options",
        set_jit_fusion_options_wrapper,
    )
    mspm.register_patch("megatron.training.training.pretrain", pretrain)
    mspm.register_patch(
        "megatron.core.transformer."
        "transformer_config.TransformerConfig.__post_init__",
        transformer_config_post_init_wrapper,
    )
    mspm.register_patch(
        "megatron.training.dist_signal_handler.get_device", get_device_wrapper
    )


patch_features()
