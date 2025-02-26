# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Input/output checkpointing."""

import os
import sys
from functools import wraps
import torch

from megatron.core import mpu, tensor_parallel, dist_checkpointing
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.training import get_args
from megatron.training.utils import (unwrap_model, print_rank_0)

from megatron.training.checkpointing import (
    get_rng_state,
    get_checkpoint_name,
    get_distributed_optimizer_checkpoint_name,
    ensure_directory_exists,
    get_checkpoint_tracker_filename,
    read_metadata,
    find_checkpoint_rank_0
)


def save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                    num_floating_point_operations_so_far, checkpointing_context=None):
    args = get_args()

    # Only rank zero of the data parallel writes to the disk.
    model = unwrap_model(model)

    ckpt_format = args.dist_ckpt_format if args.use_dist_ckpt else 'torch'
    print_rank_0('saving checkpoint at iteration {:7d} to {} in {} format'.format(
        iteration, args.save, ckpt_format))

    # Collect rng state across data parallel ranks.
    rng_state = get_rng_state(args.use_dist_ckpt)

    # Checkpoint name.
    checkpoint_name = get_checkpoint_name(args.save, iteration, return_base_dir=args.use_dist_ckpt)

    # Save distributed optimizer's custom parameter state.
    if args.use_distributed_optimizer and not args.no_save_optim and optimizer is not None and not args.use_dist_ckpt:
        optim_checkpoint_name = \
            get_distributed_optimizer_checkpoint_name(checkpoint_name)
        ensure_directory_exists(optim_checkpoint_name)
        optimizer.save_parameter_state(optim_checkpoint_name)

    async_save_request = None
    if args.async_save:
        if not args.use_dist_ckpt:
            raise NotImplementedError('Async checkpoint save not implemented for legacy checkpoints')
        elif args.dist_ckpt_format != 'torch_dist':
            raise NotImplementedError(
                f'Async checkpoint save not implemented for {args.dist_ckpt_format} distributed checkpoint format')

    # Collect args, model, RNG.
    if not torch.distributed.is_initialized() \
            or mpu.get_data_modulo_expert_parallel_rank() == 0 \
            or args.use_dist_ckpt:

        optim_sd_kwargs = {}
        if args.use_dist_ckpt and args.use_distributed_optimizer:
            optim_sd_kwargs['sharding_type'] = ('fully_sharded_bucket_space'
                                                if args.ckpt_fully_parallel_save
                                                else 'dp_zero_gather_scatter')
            print_rank_0(f'Storing distributed optimizer sharded state of type {optim_sd_kwargs["sharding_type"]}')
        state_dict = generate_state_dict(args, model, optimizer, opt_param_scheduler, rng_state,
                                         args.use_dist_ckpt, iteration, optim_sd_kwargs=optim_sd_kwargs)

        state_dict['num_floating_point_operations_so_far'] = num_floating_point_operations_so_far
        if args.use_dist_ckpt:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                ensure_directory_exists(checkpoint_name, check_parent=False)
            validate_sharding_integrity = True
            save_strategy = (checkpointing_context or {}).get('save_strategy',
                                                              get_default_save_sharded_strategy(args.dist_ckpt_format))
            if args.ckpt_fully_parallel_save:
                if checkpointing_context is not None and 'save_strategy' in checkpointing_context:
                    # Already saved once before - don't need to rerun sharding validation
                    validate_sharding_integrity = not args.ckpt_assume_constant_structure
                else:
                    save_strategy = FullyParallelSaveStrategyWrapper(save_strategy, mpu.get_data_parallel_group(
                        with_context_parallel=True),
                                                                     args.ckpt_assume_constant_structure)
            # Store save strategy for future checkpoint saves
            if checkpointing_context is not None:
                checkpointing_context['save_strategy'] = save_strategy
            async_save_request = dist_checkpointing.save(state_dict, checkpoint_name, save_strategy,
                                                         async_sharded_save=args.async_save)
        else:
            # Save.
            if args.use_ema:
                ema_state_dict = {k: v for k, v in state_dict.items() if k.startswith('ema')}
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('ema')}

            ensure_directory_exists(checkpoint_name)
            torch.save(state_dict, checkpoint_name)

            if args.use_ema:
                ema_state_dict = {k.replace('ema', 'model'): v for k, v in ema_state_dict.items()}
                torch.save(ema_state_dict, checkpoint_name + ".ema")

    if not args.async_save:
        assert async_save_request is None
        # Wait so everyone is done (necessary)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # And update the latest iteration
    if not torch.distributed.is_initialized() \
            or torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)

        def iter_finalize_fn():
            with open(tracker_filename, 'w') as f:
                f.write(str(iteration))
            print_rank_0('  successfully saved checkpoint from iteration {:7d} to {}'
                         .format(iteration, args.save))
            if args.log_progress and args.async_save:
                append_to_progress_log(f'Saved async checkpoint\tIteration: {iteration}',
                                       barrier=False)

        if args.async_save:
            assert async_save_request is not None
            async_save_request.add_finalize_fn(iter_finalize_fn)
        else:
            iter_finalize_fn()

    if args.async_save:
        schedule_async_save(async_save_request)
        print_rank_0('  scheduled an async checkpoint save at iteration {:7d} to {}' \
                     .format(iteration, args.save))

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def generate_state_dict(args, model, optimizer, opt_param_scheduler,
                        rng_state, use_dist_ckpt=False, iteration=None,
                        optim_sd_kwargs=None):
    # Arguments, iteration, and model.
    state_dict = {}
    ema_state_dict = {}
    state_dict['args'] = args
    state_dict['checkpoint_version'] = 3.0
    if iteration is not None:
        state_dict['iteration'] = iteration

    if len(model) == 1:
        state_dict['model'] = (model[0].sharded_state_dict()
                               if use_dist_ckpt else
                               model[0].state_dict_for_save_checkpoint())
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            state_dict['model%d' % i] = (
                model[i].sharded_state_dict()
                if use_dist_ckpt else
                model[i].state_dict_for_save_checkpoint())

    if args.use_ema:
        if len(model) == 1:
            state_dict['ema'] = {k: v for k, v in state_dict['model'].items() if k.startswith('ema')}
            state_dict['model'] = {k: v for k, v in state_dict['model'].items() if not k.startswith('ema')}
        else:
            for i in range(len(model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                state_dict['ema%d' % i] = {k.replace('ema.', ''): v for k, v in state_dict['model%d' % i].items() if
                                           k.startswith('ema')}
                state_dict['model%d' % i] = {k: v for k, v in state_dict['model%d' % i].items() if
                                             not k.startswith('ema')}

    # Optimizer stuff.
    if not args.no_save_optim:
        if optimizer is not None:
            state_dict['optimizer'] = (optimizer.sharded_state_dict(state_dict, **(optim_sd_kwargs or {}))
                                       if use_dist_ckpt else
                                       optimizer.state_dict())
        if opt_param_scheduler is not None:
            state_dict['opt_param_scheduler'] = \
                opt_param_scheduler.state_dict()
    # RNG states.
    if not args.no_save_rng:
        state_dict["rng_state"] = rng_state
    return state_dict


def _load_base_checkpoint(load_dir, rank0=False, sharded_state_dict=None,
                          exit_on_missing_checkpoint=False, checkpoint_step=None):
    """ Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    """
    args = get_args()

    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_dir)

    # If no tracker file, return nothing
    if not os.path.isfile(tracker_filename):
        if not rank0:
            print_rank_0('WARNING: could not find the metadata file {} '.format(
                tracker_filename))
            print_rank_0('    will not load any checkpoints and will start from '
                         'random')

        # Conditionally exit if checkpoint not found.
        if exit_on_missing_checkpoint:
            print_rank_0(">> '--exit-on-missing-checkpoint' set ... exiting. <<")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            sys.exit()

        return None, "", False

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    if checkpoint_step is not None:
        iteration = checkpoint_step
        release = False
    else:
        iteration, release = read_metadata(tracker_filename)

    # Checkpoint.
    if rank0:
        checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)
        is_dist_ckpt = checkpoint_name is not None and dist_checkpointing.check_is_distributed_checkpoint(
            checkpoint_name)
    else:
        checkpoint_name = get_checkpoint_name(load_dir, iteration, release,
                                              return_base_dir=True)
        is_dist_ckpt = dist_checkpointing.check_is_distributed_checkpoint(checkpoint_name)
        if not is_dist_ckpt:
            checkpoint_name = get_checkpoint_name(load_dir, iteration, release,
                                                  return_base_dir=False)
        dist_infix = "distributed " if is_dist_ckpt else ""
        if release:
            print_rank_0(f' loading release {dist_infix}checkpoint from {load_dir}')
        else:
            print_rank_0(f' loading {dist_infix}checkpoint from {load_dir} at iteration {iteration}')

    # Load the checkpoint.
    if is_dist_ckpt:
        if rank0:
            state_dict = dist_checkpointing.load_common_state_dict(checkpoint_name)
            return state_dict, checkpoint_name, release

        # at this point args are available
        args = get_args()
        if sharded_state_dict is None:
            assert not args.auto_detect_ckpt_format and not args.use_dist_ckpt, (
            args.auto_detect_ckpt_format, args.use_dist_ckpt)
            raise RuntimeError(
                'Detected load from a distributed checkpoint, but neither --use-dist-ckpt nor --auto-detect-ckpt-format is set.')

        load_strategy = get_default_load_sharded_strategy(checkpoint_name)
        if args.ckpt_fully_parallel_load:
            load_strategy = FullyParallelLoadStrategyWrapper(load_strategy,
                                                             mpu.get_data_parallel_group(with_context_parallel=True))
        state_dict = dist_checkpointing.load(sharded_state_dict, checkpoint_name, load_strategy)
        return state_dict, checkpoint_name, release

    try:
        state_dict = torch.load(checkpoint_name, map_location='cpu')
        try:
            args = get_args()
            if not args.use_ema:
                return state_dict, checkpoint_name, release

            len_model = sum(1 for key in state_dict if key.startswith('model'))
            ema_state_dict = torch.load(checkpoint_name + ".ema", map_location='cpu')

            if len(ema_state_dict) == 0:
                return state_dict, checkpoint_name, release

            if len_model == 1:
                ema_state_dict['model'] = {f'ema.{k}': v for k, v in ema_state_dict['model'].items()}
                state_dict['model'].update(ema_state_dict['ema'])
            else:
                for i in range(len_model):
                    ema_state_dict['model%d' % i] = {f'ema.{k}': v for k, v in ema_state_dict['model%d' % i].items()}
                    state_dict['model%d' % i].update(ema_state_dict['model%d' % i])
        except BaseException as e:
            print_rank_0('could not load the ema checkpoint, continue without ema checkpoint')
            print_rank_0(e)
            ema_state_dict = {}
    except ModuleNotFoundError:
        from megatron.legacy.fp16_deprecated import loss_scaler
        # For backward compatibility.
        if not rank0:
            print_rank_0(' > deserializing using the old code structure ...')
        sys.modules['fp16.loss_scaler'] = sys.modules[
            'megatron.legacy.fp16_deprecated.loss_scaler']
        sys.modules['megatron.fp16.loss_scaler'] = sys.modules[
            'megatron.legacy.fp16_deprecated.loss_scaler']
        sys.modules['megatron.model'] = sys.modules['megatron.legacy.model']
        state_dict = torch.load(checkpoint_name, map_location='cpu')
        sys.modules.pop('fp16.loss_scaler', None)
        sys.modules.pop('megatron.fp16.loss_scaler', None)
        sys.modules.pop('megatron.model', None)
    except BaseException as e:
        print_rank_0('could not load the checkpoint')
        print_rank_0(e)
        sys.exit()

    return state_dict, checkpoint_name, release


def save_checkpoint_ema_wrapper(func):
    @wraps(func)
    def save_checkpoint_ema(*args, **kwargs):
        model, optimizer, opt_param_scheduler = args[1:4]
        state_dict = get_ema_model(model, optimizer)
        setattr(opt_param_scheduler, 'ema_model_state_dict', state_dict)
        func(*args[:3], opt_param_scheduler, *args[4:], **kwargs)
        setattr(opt_param_scheduler, 'ema_model_state_dict', None)

    return save_checkpoint_ema


def generate_state_dict_ema_wrapper(func):
    @wraps(func)
    def generate_state_dict_ema(*args, **kwargs):
        opt_param_scheduler = args[3]
        state_dict = func(*args, **kwargs)
        if hasattr(opt_param_scheduler, 'ema_model_state_dict'):
            ema_model_state_dict = getattr(opt_param_scheduler, 'ema_model_state_dict')
            state_dict.update(ema_model_state_dict)
        return state_dict

    return generate_state_dict_ema


def get_ema_model(model, optimizer):
    state_dict = dict()
    global_args = get_args()
    use_dist_ckpt = global_args.use_dist_ckpt
    unwrapped_model = unwrap_model(model)
    unchained_optimizer = unchain_optimizer(optimizer)
    ema_optimizer_applier(unchained_optimizer)
    if len(unwrapped_model) == 1:
        state_dict['ema_model'] = (unwrapped_model[0].shared_state_dict()
                                   if use_dist_ckpt else
                                   unwrapped_model[0].state_dict_for_save_checkpoint())
        state_dict = ema_state_dict_to_cpu(state_dict, 'ema_model')
        ema_optimizer_restore(unchained_optimizer)
        return state_dict
    for sub_model in unwrapped_model:
        sub_model_idx = unwrapped_model.index(sub_model)
        mpu.set_virtual_pipeline_model_parallel_rank(sub_model_idx)
        state_dict['ema_model%d' % sub_model_idx] = (
            sub_model.sharded_state_dict()
            if use_dist_ckpt else
            sub_model.state_dict_for_save_checkpoint())
        state_dict = ema_state_dict_to_cpu(state_dict, 'ema_model%d' % sub_model_idx)
    ema_optimizer_restore(unchained_optimizer)
    return state_dict


def unchain_optimizer(chained_optimizer):
    if isinstance(chained_optimizer, ChainedOptimizer):
        return chained_optimizer.chained_optimizers
    return [chained_optimizer]


def ema_optimizer_applier(unchained_optimizer):
    for optim in unchained_optimizer:
        optim.optimizer.store(optim.optimizer.param_groups)
        optim.optimizer.copy_to()
        param_sync(optim)


def ema_optimizer_restore(unchained_optimizer):
    for optim in unchained_optimizer:
        optim.optimizer.restore(optim.optimizer.param_groups)
        param_sync(optim)
    torch.distributed.barrier()
    for optim in unchained_optimizer:
        optim.update_successful = False


def param_sync(optim):
    if hasattr(optim, "_copy_main_params_to_model_params"):
        optim._copy_main_params_to_model_params()
    if hasattr(optim, "_reset_metadata_and_sync_gather_all_model_params"):
        optim.update_successful = True
        optim._reset_metadata_and_sync_gather_all_model_params(force_sync=True)


def ema_state_dict_to_cpu(state_dict, ema_key):
    for k, v in state_dict[ema_key].items():
        if not torch.is_tensor(v):
            continue
        new_v = v.detach().cpu().clone()
        state_dict[ema_key][k] = new_v
    return state_dict
