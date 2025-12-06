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

import sys
import contextlib
import random

import torch
import numpy as np
from megatron.core import mpu, tensor_parallel
from megatron.training import get_args, ft_integration, wandb_utils
from megatron.training.utils import (
    unwrap_model,
    print_rank_0,
    is_last_rank
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.training.checkpointing import (
    checkpoint_exists,
    CheckpointType,
    fix_fp8_params_lose_precision_when_loading_dist_ckpt,
    set_checkpoint_version,
    _to_dtensor,
    check_checkpoint_args,
    get_checkpoint_version,
    fix_query_key_value_ordering
)
from megatron.core.num_microbatches_calculator import update_num_microbatches
# [ModelOpt]: Import
try:
    from modelopt.torch.opt.plugins import (
        save_modelopt_state,
        save_sharded_modelopt_state,
        restore_modelopt_state,
        restore_sharded_modelopt_state,
    )
    has_nvidia_modelopt = True
except Exception:
    has_nvidia_modelopt = False

from megatron.training.checkpointing import (
    get_rng_state,
    get_checkpoint_name,
    get_distributed_optimizer_checkpoint_name,
    get_checkpoint_tracker_filename,
    read_metadata,
)
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict


def generate_state_dict(args, model, optimizer, opt_param_scheduler,
                        rng_state, use_dist_ckpt=False, iteration=None,
                        optim_sd_kwargs=None, rerun_state=None):
    """Generate a state dict from given model, optimizer, scheduler, rng state and others.

    Note: use_dist_ckpt is deprecated and not used. Will be removed soon.
    """

    # Arguments, iteration, and model.
    state_dict = {}
    state_dict['args'] = args
    state_dict['checkpoint_version'] = 3.0
    if iteration is not None:
        state_dict['iteration'] = iteration

    for i in range(len(model)):
        key = "model"
        if len(model) > 1:
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            key = f"model{i}"

        model_sd = None
        if args.ckpt_format == "torch_dist":
            model_sd = model[i].sharded_state_dict()
        else:   # torch, torch_dcp
            model_sd = model[i].state_dict_for_save_checkpoint()

        state_dict[key] = model_sd

    # Optimizer stuff.
    if not args.no_save_optim:
        if optimizer is not None and not optimizer.is_stub_optimizer:
            optimizer_sd = None

            if args.ckpt_format == "torch_dist":
                optimizer_sd = optimizer.sharded_state_dict(state_dict, **(optim_sd_kwargs or {}))
            elif args.ckpt_format == "torch_dcp":
                optimizer_sd = get_optimizer_state_dict(model[0], [opt.optimizer for opt in optimizer.chained_optimizers])
            else:
                optimizer_sd = optimizer.state_dict()

            state_dict['optimizer'] = optimizer_sd

        if opt_param_scheduler is not None:
            state_dict['opt_param_scheduler'] = \
                opt_param_scheduler.state_dict()

    # Rerun state
    state_dict['rerun_state_machine'] = rerun_state

    # RNG states.
    if not args.no_save_rng:
        state_dict["rng_state"] = rng_state
    return state_dict


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, load_arg='load', strict=True,
                    checkpointing_context=None, skip_load_to_model_and_opt=False):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    skip_load_to_model_and_opt (bool): whether to call `load_state_dict`
        for :attr:`model` and :attr:`optimizer`. In case of running FSDP2 with mcore distributed
        checkpointing, the tensors are already loaded in-place by `_load_base_checkpoint`.
    """
    from megatron.training.checkpointing import (
        _load_base_checkpoint,
        generate_state_dict
    )

    args = get_args()
    load_dir = getattr(args, load_arg)

    # Finetuning directories
    pretrained_dir = getattr(args, 'pretrained_checkpoint', None)
    if pretrained_dir is not None and not checkpoint_exists(load_dir):
        print_rank_0(
            f'Checkpoint file not found in load directory {load_dir} attempting to finetune with checkpoint in {pretrained_dir}'
        )
        load_dir = pretrained_dir
        if not checkpoint_exists(load_dir):
            raise FileNotFoundError("No checkpoint found in load directory or pretrained directory")
        args.finetune = True

    model = unwrap_model(ddp_model)

    ckpt_format = args.ckpt_format
    if args.auto_detect_ckpt_format or ckpt_format == "torch_dist":
        state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
            load_dir,
            args,
            rank0=True,
            checkpointing_context=checkpointing_context,
        )

        ckpt_format = None
        if ckpt_type == CheckpointType.TORCH_DCP:
            ckpt_format = "torch_dcp"
        elif ckpt_type == CheckpointType.LEGACY:
            ckpt_format = "torch"
        elif ckpt_type in [CheckpointType.LOCAL, CheckpointType.GLOBAL]:
            ckpt_format = "torch_dist"
        elif ckpt_type is None:
            pass  # Not loaded.
        else:
            raise NotImplementedError(f"checkpoint format {ckpt_format} not supported")

    load_kwargs = {}
    if ckpt_format == "torch_dist":
        ckpt_tp_pp = (
            state_dict['args'].tensor_model_parallel_size,
            state_dict['args'].pipeline_model_parallel_size,
            getattr(state_dict['args'], 'encoder_tensor_model_parallel_size', 0),
            getattr(state_dict['args'], 'encoder_pipeline_model_parallel_size', 0),
        )
        run_tp_pp = (
            args.tensor_model_parallel_size,
            args.pipeline_model_parallel_size,
            getattr(args, 'encoder_tensor_model_parallel_size', 0),
            getattr(args, 'encoder_pipeline_model_parallel_size', 0),
        )
        mismatch_msg = "(TP, PP, encoder TP, encoder PP) mismatch after resume ({} vs {} from checkpoint)".format(
            run_tp_pp, ckpt_tp_pp
        )

        # Determine if RNG state will be loaded
        if (ckpt_tp_pp == run_tp_pp and not release and not args.finetune and not args.no_load_rng
                and not getattr(state_dict['args'], 'no_save_rng', False)):
            gen_sd_rng_state = get_rng_state(args.ckpt_format)  # we can load the rng state
        else:
            gen_sd_rng_state = None
            if ckpt_tp_pp != run_tp_pp:
                print_rank_0("{}: RNG state will be ignored".format(mismatch_msg))

        optim_sd_kwargs = dict(is_loading=True)
        # Determine if optimizer state will be loaded
        if (not release and not args.finetune and not args.no_load_optim
                and not getattr(state_dict['args'], 'no_save_optim', False)):
            gen_sd_optim = optimizer
            gen_sd_opt_param_scheduler = opt_param_scheduler

            if args.use_distributed_optimizer:
                optim_sd_kwargs['sharding_type'] = ('fully_sharded_model_space'
                                                    if getattr(state_dict['args'], 'ckpt_fully_parallel_save', False)
                                                    else 'dp_zero_gather_scatter')
                # This is for backwards-compatibility. Can be removed once 'fully_sharded_bucket_space' loading is removed
                for maybe_dist_opt_optim_state in (state_dict['optimizer'], *state_dict['optimizer'].values()):
                    if 'param_state_sharding_type' in maybe_dist_opt_optim_state:
                        if maybe_dist_opt_optim_state['param_state_sharding_type'] == 'fully_sharded_bucket_space':
                            print_rank_0(
                                'Detected deprecated `fully_sharded_bucket_space` DistributedOptimizer checkpoint format')
                            optim_sd_kwargs['sharding_type'] = maybe_dist_opt_optim_state['param_state_sharding_type']
                        break

                if ckpt_tp_pp != run_tp_pp and optim_sd_kwargs['sharding_type'] != 'fully_sharded_model_space':
                    raise RuntimeError(
                        f"{mismatch_msg}: not supported for DistributedOptimizer with sharding type {optim_sd_kwargs['sharding_type']}."
                        f" Please use `--ckpt-fully-parallel-save` flag during checkpoint saving.")
        else:
            gen_sd_optim = None
            gen_sd_opt_param_scheduler = None

        # Determine if rerun state will be loaded
        if (
                ckpt_tp_pp == run_tp_pp
                and not release
                and not args.finetune
                and 'rerun_state_machine' in state_dict
        ):
            rerun_state_machine = get_rerun_state_machine()
            gen_sd_rerun_state = rerun_state_machine.state_dict(
                data_iterator=None, ckpt_format=ckpt_format,
            )
        else:
            gen_sd_rerun_state = None
            if ckpt_tp_pp != run_tp_pp:
                print_rank_0("{}: Rerun state will be ignored".format(mismatch_msg))

        # [ModelOpt]: IMPORTANT! Restoring modelopt_state (sharded or not) must be performed
        # after the model instance has been created and before _load_base_checkpoint is called.
        if has_nvidia_modelopt:
            if ckpt_type == CheckpointType.LOCAL:
                print_rank_0('WARNING: Local checkpointing does not support nvidia_modelopt.')
            elif ckpt_type == CheckpointType.GLOBAL:
                restore_modelopt_state(model, state_dict)
            else:
                restore_sharded_modelopt_state(model, checkpoint_name)

        # [ModelOpt]: Initial loading from non-resume sharded checkpoint to a Distillation Model
        # will result in key mismatch with loss modules potentially containing parameters, since
        # it requires generating a state_dict before loading. Here we hide those modules if present.
        with contextlib.ExitStack() as stack:  # Allows multiple context managers for each model shard
            if args.finetune and hasattr(model[0], "hide_loss_modules"):
                for m in model:
                    stack.enter_context(m.hide_loss_modules())
            load_kwargs['sharded_state_dict'] = generate_state_dict(
                args, model, gen_sd_optim, gen_sd_opt_param_scheduler, gen_sd_rng_state,
                optim_sd_kwargs=optim_sd_kwargs, rerun_state=gen_sd_rerun_state
            )

        # When "--fp8-param-gather" is disabled, this function doesn't modify anything.
        fix_fp8_params_lose_precision_when_loading_dist_ckpt(load_kwargs['sharded_state_dict'])
    elif args.ckpt_format == "torch_dcp":
        # FIX: Conditional state dict construction for Megatron compatibility
        # Previously all fields (optimizer, RNG) were always included regardless of flags,
        # now they are conditionally added based on --no_load_optim and --no_load_rng flags
        model_sd = model[0].state_dict()
        rerun_state_machine = get_rerun_state_machine()
        rerun_state = rerun_state_machine.state_dict(
            data_iterator=[], ckpt_format=args.ckpt_format,
        )
        # Initialize sharded state dictionary with mandatory components
        sharded_state_dict = {
            "model": model_sd,
            "args": None,
            "iteration": 1,
            "rerun_state_machine": rerun_state,
            "checkpoint_version": None,
            "num_floating_point_operations_so_far": 0,
        }

        if not args.no_load_optim:
            if optimizer is not None and not optimizer.is_stub_optimizer:
                # FIX: Conditional state dict construction for Megatron compatibility
                # Previously only,
                # now they are conditionally added based on --no_load_optim and --no_load_rng flags
                optimizer_sd = get_optimizer_state_dict(model[0], [opt.optimizer for opt in optimizer.chained_optimizers])
                sharded_state_dict.update({"optimizer": optimizer_sd})
            if opt_param_scheduler is not None:
                sharded_state_dict.update({"opt_param_scheduler": opt_param_scheduler.state_dict()})

        if not args.no_load_rng:
            sharded_state_dict.update({"rng_state": get_rng_state(args.ckpt_format)})

        load_kwargs["sharded_state_dict"] = sharded_state_dict

    state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
        load_dir, args, rank0=False, checkpointing_context=checkpointing_context,
        **load_kwargs
    )

    # Checkpoint not loaded.
    if state_dict is None:
        # Iteration and num_floating_point_operations_so_far default to 0.
        return 0, 0

    # Set checkpoint version.
    set_checkpoint_version(state_dict.get('checkpoint_version', 0))

    # Convert to regular torch tensor to DTensor.
    if ckpt_type == CheckpointType.LEGACY and args.ckpt_format == "torch_dcp":
        dtensor_state_dict = _to_dtensor(ddp_model, state_dict["model"])
        state_dict["model"] = dtensor_state_dict

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = state_dict['iteration']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = state_dict['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but unable to load '
                             'iteration from checkpoint {}, exiting'.format(checkpoint_name))
                sys.exit()
    num_floating_point_operations_so_far = state_dict.get('num_floating_point_operations_so_far', 0)

    # Check arguments.
    assert args.consumed_train_samples == 0
    assert args.skipped_train_samples == 0
    assert args.consumed_valid_samples == 0
    if 'args' in state_dict and not args.finetune:
        checkpoint_args = state_dict['args']
        check_checkpoint_args(checkpoint_args)
        args.consumed_train_samples = getattr(checkpoint_args,
                                              'consumed_train_samples', 0)
        args.skipped_train_samples = getattr(checkpoint_args,
                                             'skipped_train_samples', 0)
        update_num_microbatches(consumed_samples=args.consumed_train_samples, verbose=True)
        args.consumed_valid_samples = getattr(checkpoint_args,
                                              'consumed_valid_samples', 0)
    else:
        print_rank_0('could not find arguments in the checkpoint ...')

    # Model.
    strict = False if args.retro_add_retriever else strict
    if not skip_load_to_model_and_opt:
        if len(ddp_model) == 1:
            ddp_model[0].load_state_dict(state_dict['model'], strict=strict)
        else:
            for i in range(len(ddp_model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                ddp_model[i].load_state_dict(state_dict['model%d' % i], strict=strict)

    # Fix up query/key/value matrix ordering if needed.
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f' checkpoint version {checkpoint_version}')
    fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            # Load state dict.
            if not skip_load_to_model_and_opt and optimizer is not None and not optimizer.is_stub_optimizer:
                optimizer.load_state_dict(state_dict['optimizer'])

            # Load distributed optimizer's custom parameter state.
            # For distributed checkpoint it's already loaded in load_state_dict above
            is_torch_dist = ckpt_format == "torch_dist"
            if args.use_distributed_optimizer and not is_torch_dist:
                # NOTE: this is a manual read of the tracker file.
                # This code should not be reached when reading from a non_persistent checkpoint
                assert not is_torch_dist
                tracker_filename = get_checkpoint_tracker_filename(load_dir)
                iteration, release = read_metadata(tracker_filename)
                model_checkpoint_name = \
                    get_checkpoint_name(load_dir, iteration, release)
                optim_checkpoint_name = \
                    get_distributed_optimizer_checkpoint_name(
                        model_checkpoint_name)
                optimizer.load_parameter_state(optim_checkpoint_name,
                                               update_legacy_format=args.ckpt_convert_update_legacy_dist_opt_format)

            # Load scheduler.
            if opt_param_scheduler is not None:
                if 'lr_scheduler' in state_dict:  # backward compatbility
                    opt_param_scheduler.load_state_dict(state_dict['lr_scheduler'])
                else:
                    opt_param_scheduler.load_state_dict(state_dict['opt_param_scheduler'])
        except KeyError as e:
            print_rank_0('Unable to load optimizer from checkpoint {}. '
                         'Specify --no-load-optim or --finetune to prevent '
                         'attempting to load the optimizer state, '
                         'exiting ...'.format(checkpoint_name))
            raise e
    else:
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()

    # rerun state
    try:
        if 'rerun_state_machine' in state_dict:
            get_rerun_state_machine().load_state_dict(state_dict['rerun_state_machine'])
    except Exception as e:
        print(f"Unable to restore RerunMachine from checkpoint: {e}")
        sys.exit()

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            if 'rng_state' in state_dict:
                # access rng_state for data parallel rank
                if args.data_parallel_random_init:
                    rng_state = state_dict['rng_state'][mpu.get_data_parallel_rank()]
                else:
                    rng_state = state_dict['rng_state'][0]
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
        except KeyError:
            print_rank_0('Unable to load rng state from checkpoint {}. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the rng state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()

    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(f'  successfully loaded checkpoint from {load_dir} '
                 f'[ t {mpu.get_tensor_model_parallel_rank() + 1}/{mpu.get_tensor_model_parallel_world_size()}, '
                 f'p {mpu.get_pipeline_model_parallel_rank() + 1}/{mpu.get_pipeline_model_parallel_world_size()} ] '
                 f'at iteration {iteration}')

    # Additional callback for wandb (last rank)
    if not torch.distributed.is_initialized() \
            or is_last_rank():
        wandb_utils.on_load_checkpoint_success(checkpoint_name, load_dir)

    torch.cuda.empty_cache()

    if iteration > 0:
        # Notify FT that a checkpoint was loaded.
        is_local_chkpt = (ckpt_type == CheckpointType.LOCAL)
        ft_integration.on_checkpoint_loaded(is_local_chkpt=is_local_chkpt)

    return iteration, num_floating_point_operations_so_far