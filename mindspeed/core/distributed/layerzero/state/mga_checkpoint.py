# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

import os
import random
import warnings
import sys
import numpy as np

import torch
import torch.distributed as dist
from megatron.training.checkpointing import get_rng_state
from megatron.training.global_vars import get_args
from megatron.training.utils import print_rank_0
from megatron.core import mpu, tensor_parallel

from .state_dict import shard_state_dict, clean_ignored_modules, use_zero1_params
from .optim_state import _shard_optim_state_dict

PARALLE_STATE_KAY = "parallel_state"
MODEL_KEY = "model"
RNG_STATE_KEY = "rng_state"
SHRAD_KEY = "shard_state_dict"
EMA_MODEL_KEY = "ema_model"
OPTIM_STATE_KEY = "optimizer"
OPTIM_INFO_KEY = "optimizer_param_key_to_fqn"
OPTIM_SCHEDULER_KEY = "opt_param_scheduler"
LR_SCHEDULER_KEY = "lr_scheduler"


def save_checkpoint(iteration, model, optimizer, opt_param_scheduler, num_floating_point_operations_so_far=None, checkpointing_context=None,
                    pipeline_rank=None, expert_rank=None, tensor_rank=None, pipeline_parallel=None, expert_parallel=None):
    """Save a model checkpoint.

    Checkpointing context is used to persist some checkpointing state
    throughout a single job. Must be initialized externally (not used if None).
    """
    args = get_args()
    if not hasattr(args, "save"):
        setattr(args, "save", "ckpt")
    print_rank_0('saving checkpoint at iteration {:7d} to {} '.format(
        iteration, args.save))
    rng_state = get_rng_state(False)
    checkpoint_name = get_checkpoint_name(args.save, iteration, release=False)

    # Collect args, model, RNG.
    state_dict = generate_state_dict(args, model, optimizer, opt_param_scheduler, rng_state,
                                     False, iteration)
    state_dict[PARALLE_STATE_KAY] = generate_3D_parallel_state()
    state_dict['num_floating_point_operations_so_far'] = num_floating_point_operations_so_far

    ensure_directory_exists(checkpoint_name)
    print_rank_0(f"Start Saving to {checkpoint_name}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    torch.save(state_dict, checkpoint_name)

    if dist.is_initialized():
        dist.barrier()


def generate_3D_parallel_state():
    # Ensure the distributed environment is initialized
    if not dist.is_initialized():
        raise RuntimeError("Distributed environment is not initialized.")

    # Ensure Megatron's parallel utilities are initialized
    if not mpu.is_initialized():
        raise RuntimeError(
            "Megatron's parallel utilities are not initialized.")

    # Get global rank
    global_rank = dist.get_rank()
    # Get tensor parallel rank
    tp_rank = mpu.get_tensor_model_parallel_rank()
    # Get pipeline parallel rank
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    # Get data parallel rank
    dp_rank = mpu.get_data_parallel_rank()
    # Get tensor parallel degree
    tp_degree = mpu.get_tensor_model_parallel_world_size()
    # Get pipeline parallel degree
    pp_degree = mpu.get_pipeline_model_parallel_world_size()
    # Get data parallel degree
    dp_degree = mpu.get_data_parallel_world_size()

    # Assemble the dictionary
    parallel_state = {
        'tp_rank': tp_rank,
        'pp_rank': pp_rank,
        'dp_rank': dp_rank,
        'tp_degree': tp_degree,
        'pp_degree': pp_degree,
        'dp_degree': dp_degree,
        'global_rank': global_rank
    }

    return parallel_state


def generate_state_dict(args, model, optimizer, opt_param_scheduler,
                        rng_state, use_dist_ckpt=False, iteration=None,
                        optim_sd_kwargs=None):
    # Arguments, iteration, and model.
    state_dict = {}
    state_dict['args'] = args
    state_dict['checkpoint_version'] = 3.0
    if iteration is not None:
        state_dict['iteration'] = iteration

    if not len(model) == 1:
        raise ValueError(f"Only single model is supported, VPP not supported")
    use_zero1_params(model[0])
    state_dict[MODEL_KEY] = clean_ignored_modules(
        model[0], model[0].state_dict())
    state_dict[SHRAD_KEY] = shard_state_dict(model[0], state_dict[MODEL_KEY])

    # Optimizer stuff.
    if not args.no_save_optim:
        if optimizer is not None:
            state_dict[OPTIM_STATE_KEY] = optimizer.state_dict()
            state_dict[OPTIM_INFO_KEY] = _shard_optim_state_dict(
                model[0], optimizer.optimizer, state_dict[OPTIM_STATE_KEY])
            if getattr(args, "optimizer_selection", None) == 'fused_ema_adamw':
                try:
                    ema_optimizer_applier(optimizer)
                    state_dict[EMA_MODEL_KEY] = clean_ignored_modules(
                        model[0], model[0].state_dict())
                    state_dict = ema_state_dict_to_cpu(
                        state_dict, EMA_MODEL_KEY)
                    ema_optimizer_restore(optimizer)
                    print_rank_0("Ema model successful saved in state_dict")
                except KeyError:
                    warnings.warn(
                        f"ema_optimizer_applier failed with KeyError, ema_model not saved")
        if opt_param_scheduler is not None:
            state_dict[OPTIM_SCHEDULER_KEY] = \
                opt_param_scheduler.state_dict()
    # RNG states.
    if not args.no_save_rng:
        state_dict[RNG_STATE_KEY] = rng_state
    return state_dict


def get_checkpoint_name(checkpoints_path, iteration, release=False):
    """Determine the directory name for this rank's checkpoint."""
    if checkpoints_path is None:
        raise ValueError("checkpoints_path cannot be None")
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)
    common_path = os.path.join(checkpoints_path, directory)
    global_rank = dist.get_rank()
    return os.path.join(common_path, f"model_{global_rank}.pt")


def ensure_directory_exists(filename, check_parent=True):
    """Build filename's path if it does not already exists."""
    if filename is None:
        raise AssertionError(f"Got {filename} filename")
    dirname = os.path.dirname(filename) if check_parent else filename
    os.makedirs(dirname, exist_ok=True)


def load_layerzero_checkpoint(models, ckpt_dir, optimizer=None, opt_param_scheduler=None):
    if ckpt_dir is None:
        raise AssertionError(f"Got {ckpt_dir} filename")
    if len(models) != 1:
        raise ValueError(f"VPP is not supported by layerzero currently")
    rank = dist.get_rank()
    sd_file = os.path.join(ckpt_dir, f"model_{rank}.pt")
    if not os.path.exists(sd_file):
        raise FileNotFoundError(
            f"No checkpoint found in load directory or pretrained directory: no such file {sd_file}")
    args = get_args()
    state_dict = torch.load(sd_file)
    for i in range(len(models)):
        models[i].load_state_dict(state_dict[MODEL_KEY], strict=False)
        if not args.finetune and not args.no_load_optim:
            try:
                # Load state dict.
                if optimizer is not None:
                    optimizer.load_state_dict(state_dict[OPTIM_STATE_KEY])
                if opt_param_scheduler is not None:
                    if LR_SCHEDULER_KEY in state_dict:  # backward compatbility
                        opt_param_scheduler.load_state_dict(
                            state_dict[LR_SCHEDULER_KEY])
                    else:
                        opt_param_scheduler.load_state_dict(
                            state_dict[OPTIM_SCHEDULER_KEY])
            except KeyError as e:
                raise RuntimeError('Unable to load optimizer from checkpoint {}. '
                             'Specify --no-load-optim or --finetune to prevent '
                             'attempting to load the optimizer state, '
                             'exiting ...'.format(ckpt_dir)) from e
    args.num_floating_point_operations_so_far = state_dict.get(
        'num_floating_point_operations_so_far', 0)
    if args.finetune:
        iteration = 0
    else:
        try:
            iteration = state_dict['iteration']
        except KeyError:
            iteration = 0
    args.iteration = iteration

    # Check arguments.
    update_consumed_samples(args, state_dict)
    # rng states.
    resume_rng_states(args, state_dict)

    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(f'  successfully loaded checkpoint from {ckpt_dir} '
                 f'[ t {mpu.get_tensor_model_parallel_rank()}, '
                 f'p {mpu.get_pipeline_model_parallel_rank()} ] '
                 f'at iteration {iteration}')
    return args.iteration, args.num_floating_point_operations_so_far


def update_consumed_samples(args, state_dict):
    if 'args' in state_dict and not args.finetune:
        checkpoint_args = state_dict['args']
        args.consumed_train_samples = getattr(checkpoint_args,
                                              'consumed_train_samples', 0)
        try:
            from megatron.core.num_microbatches_calculator import update_num_microbatches
            update_num_microbatches(
                consumed_samples=args.consumed_train_samples)
        except ImportError:
            pass
        args.consumed_valid_samples = getattr(checkpoint_args,
                                              'consumed_valid_samples', 0)
    else:
        print_rank_0('could not find arguments in the checkpoint ...')


def resume_rng_states(args, state_dict):
    if not args.finetune and not args.no_load_rng:
        try:
            if RNG_STATE_KEY in state_dict:
                # access rng_state for data parallel rank
                if args.data_parallel_random_init:
                    rng_state = state_dict[RNG_STATE_KEY][mpu.get_data_parallel_rank(
                    )]
                else:
                    rng_state = state_dict[RNG_STATE_KEY][0]
                random.setstate(rng_state['random_rng_state'])
                np.random.set_state(rng_state['np_rng_state'])
                torch.set_rng_state(rng_state['torch_rng_state'])
                torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
                # Check for empty states array
                if not rng_state['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    rng_state['rng_tracker_states'])
            else:  # backward compatability
                random.setstate(state_dict['random_rng_state'])
                np.random.set_state(state_dict['np_rng_state'])
                torch.set_rng_state(state_dict['torch_rng_state'])
                torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
                # Check for empty states array
                if not state_dict['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    state_dict['rng_tracker_states'])
        except KeyError as e:
            raise RuntimeError('Unable to load rng state from checkpoint  '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the rng state, '
                         'exiting ...') from e


def ema_optimizer_applier(optimizer):
    if hasattr(optimizer, "optimizer"):
        optimizer.optimizer.store(optimizer.optimizer.param_groups)
        optimizer.optimizer.copy_to()
        return


def ema_optimizer_restore(optimizer):
    if hasattr(optimizer, "optimizer"):
        optimizer.optimizer.restore(optimizer.optimizer.param_groups)
        return


def ema_state_dict_to_cpu(state_dict, ema_key):
    for k, v in state_dict[ema_key].items():
        if not torch.is_tensor(v):
            continue
        new_v = v.detach().cpu().clone()
        state_dict[ema_key][k] = new_v
    return state_dict
