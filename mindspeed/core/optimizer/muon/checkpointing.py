# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# pylint: skip-file

import os
from functools import wraps

from megatron.core import mpu
from megatron.training import get_args
from megatron.training.checkpointing import ensure_directory_exists, get_checkpoint_name


_CHECKPOINT_NAME = ""
_RELEASE = False


def save_checkpoint_layer_wise_optimizer_wrapper(func):
    @wraps(func)
    def save_checkpoint_layer_wise_optimizer(*func_args, **kwargs):
        args = get_args()
        iteration = kwargs["iteration"] if "iteration" in kwargs else func_args[0]
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else func_args[2]
        checkpoint_name = get_checkpoint_name(
            args.save, iteration, return_base_dir=getattr(args, "use_dist_ckpt", False)
        )
        ckpt_format = getattr(args, "ckpt_format", None)
        if ckpt_format is None:
            ckpt_format = (
                "torch" if not getattr(args, "use_dist_ckpt", False) else getattr(args, "dist_ckpt_format", None)
            )

        # LayerWiseDistributedOptimizer save optimizer state to file on different ranks
        if getattr(args, "use_layer_wise_distributed_optimizer", False) and ckpt_format == 'torch':
            dp_rank = mpu.get_data_parallel_rank()
            optim_checkpoint_name = os.path.join(os.path.dirname(checkpoint_name), f"layer_wise_optimizer_{dp_rank}.pt")
            ensure_directory_exists(optim_checkpoint_name)
            if optimizer is not None and not getattr(optimizer, "is_stub_optimizer", False):
                optimizer.save_state_dict_to_file(optim_checkpoint_name)

        return func(*func_args, **kwargs)

    return save_checkpoint_layer_wise_optimizer


def load_base_checkpoint_layer_wise_optimizer_wrapper(func):
    @wraps(func)
    def load_base_checkpoint_layer_wise_optimizer(*args, **kwargs):
        global _CHECKPOINT_NAME, _RELEASE
        result = func(*args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 3:
            _CHECKPOINT_NAME = result[1]
            _RELEASE = result[2]
        return result

    return load_base_checkpoint_layer_wise_optimizer


def load_checkpoint_layer_wise_optimizer_wrapper(func):
    @wraps(func)
    def load_checkpoint_layer_wise_optimizer(*func_args, **kwargs):
        global _CHECKPOINT_NAME, _RELEASE
        args = get_args()
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else func_args[1]
        ckpt_format = getattr(args, "ckpt_format", None)
        if ckpt_format is None:
            ckpt_format = (
                "torch" if not getattr(args, "use_dist_ckpt", False) else getattr(args, "dist_ckpt_format", None)
            )

        if (
            getattr(args, "use_layer_wise_distributed_optimizer", False)
            and ckpt_format == 'torch'
            and optimizer is not None
            and not getattr(optimizer, "is_stub_optimizer", False)
            and not getattr(args, "no_load_optim", False)
            and not getattr(args, "finetune", False)
        ):
            _CHECKPOINT_NAME = ""
            _RELEASE = False
            optimizer_load_state_dict = optimizer.load_state_dict

            def load_state_dict(*_args, **_kwargs):
                return None

            optimizer.load_state_dict = load_state_dict
            try:
                result = func(*func_args, **kwargs)
            finally:
                optimizer.load_state_dict = optimizer_load_state_dict

            if _CHECKPOINT_NAME and not _RELEASE:
                # LayerWiseDistributedOptimizer load optimizer state from file on different ranks
                dp_rank = mpu.get_data_parallel_rank()
                optim_checkpoint_name = os.path.join(
                    os.path.dirname(_CHECKPOINT_NAME), f"layer_wise_optimizer_{dp_rank}.pt"
                )
                optimizer.load_state_dict_from_file(optim_checkpoint_name)
            return result

        return func(*func_args, **kwargs)

    return load_checkpoint_layer_wise_optimizer
