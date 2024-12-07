import os
import sys

from functools import wraps
import torch

from megatron.core import mpu, tensor_parallel, dist_checkpointing
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



def generate_state_dict_ema_wrapper(func):
    @wraps(func)
    def generate_state_dict_ema(*args, **kwargs):
        state_dict = func(*args, **kwargs)
        model = args[1]
        optimizer = args[2]
        use_dist_ckpt = args[5]
        ema_optimizer_applier(optimizer)
        dtype = torch.float32
        if len(model) == 1:
            state_dict['ema_model'] = (model[0].shared_state_dict()
                                       if use_dist_ckpt else
                                       model[0].state_dict_for_save_checkpoint())
            state_dict = ema_state_dict_dtype_conversion(state_dict, 'ema_model', dtype)
            return state_dict
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            state_dict['ema_model%d' % i] = (
                model[i].sharded_state_dict()
                if use_dist_ckpt else
                model[i].state_dict_for_save_checkpoint())
            state_dict = ema_state_dict_dtype_conversion(state_dict, 'ema_model%d' % i, dtype)
        return state_dict

    return generate_state_dict_ema


def ema_optimizer_applier(chained_optimizer):
    if hasattr(chained_optimizer, "chained_optimizers"):
        for optim in chained_optimizer.chained_optimizers:
            optim.optimizer.copy_to()
        return
    if hasattr(chained_optimizer, "optimizer"):
        chained_optimizer.optimizer.copy_to()
        return


def ema_state_dict_dtype_conversion(state_dict, ema_key, dtype):
    for k, v in state_dict[ema_key].items():
        if not torch.is_tensor(v):
            continue
        new_v = v.clone().to(dtype)
        state_dict[ema_key][k] = new_v
    return state_dict