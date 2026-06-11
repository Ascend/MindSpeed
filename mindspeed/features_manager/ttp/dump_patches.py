# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# pylint: disable=cyclic-import
# Reason: apply_dump_patches() does a lazy import from adaptor (inside the
# function body), which never executes at module load time. The apparent
# cycle through adaptor→replica_optimizer→dump_save→dump_patches is a
# static analysis false positive.
"""Dump temporary patches — applied by dump_save.py before dump checkpoint.

apply/revert is called explicitly by dump_save.py at the right time.

Lifecycle:
  apply_dump_patches()    ← dump_save.py 调用 (dump 前)
  (process exits — no revert needed)

All patch targets are on Megatron code. Zero PyTorch patches.
"""

import logging

from mindspeed.patch_utils import Patch

logger = logging.getLogger(__name__)


def apply_dump_patches():
    """Apply dump patches. Called by dump_save.py before dump checkpoint."""
    from mindspeed.core.ttp.adaptor import (
        patched_save_preprocess_wrapper,
        patched_apply_saving_parallelization_wrapper,
        patched_fp_save_wrapper_init_wrapper,
        patched_sharded_tensor_to_torch_sharded_tensor_wrapper,
        patched_mcore_create_global_plan_wrapper,
    )

    dump_patch_specs = [
        ('megatron.core.dist_checkpointing.state_dict_utils.save_preprocess', patched_save_preprocess_wrapper),
        ('megatron.core.dist_checkpointing.serialization.save_preprocess', patched_save_preprocess_wrapper),
        (
            'megatron.core.dist_checkpointing.strategies.fully_parallel.'
            'FullyParallelSaveStrategyWrapper.apply_saving_parallelization',
            patched_apply_saving_parallelization_wrapper,
        ),
        (
            'megatron.core.dist_checkpointing.strategies.torch.sharded_tensor_to_torch_sharded_tensor',
            patched_sharded_tensor_to_torch_sharded_tensor_wrapper,
        ),
        (
            'megatron.core.dist_checkpointing.strategies.torch.MCoreSavePlanner.create_global_plan',
            patched_mcore_create_global_plan_wrapper,
        ),
        (
            'megatron.core.dist_checkpointing.strategies.fully_parallel.FullyParallelSaveStrategyWrapper.__init__',
            patched_fp_save_wrapper_init_wrapper,
        ),
    ]

    for target, func in dump_patch_specs:
        try:
            Patch(target, func, create_dummy=False).apply_patch()
        except Exception:
            logger.warning("[TTP] Failed to apply dump patch %s", target, exc_info=True)
