import os
import sys

from functools import wraps
import torch

from megatron.core import mpu
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.training import get_args
from megatron.training.utils import unwrap_model



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