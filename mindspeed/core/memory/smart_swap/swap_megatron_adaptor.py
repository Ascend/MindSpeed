# Copyright 2025 Huawei Technologies Co., Ltd.
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
from functools import wraps

from megatron.training.training import get_num_microbatches
from megatron.training import get_args

from .swap_manager import SwapManager


def megatron_get_optimizer_tensors_fcn(optimizer):
    results = []
    optimizers = [optimizer] if not hasattr(optimizer, "chained_optimizers") else optimizer.chained_optimizers
    for optimizer in optimizers:
        for group in optimizer.optimizer.param_groups:
            amsgrad = group["amsgrad"]
            for p in group["params"]:
                if p is None:
                    continue
                results.append(p.data)

                state = optimizer.optimizer.state[p]
                if len(state) > 0:
                    results.append(state["exp_avg"])
                    results.append(state["exp_avg_sq"])
                    if amsgrad:
                        results.append(state["max_exp_avg_sq"])

    return results


def megatron_get_shared_tensors_fcn(models):
    results = []
    for model in models:
        # SHARED_MEMORY
        if model.buffers is not None:
            for buffer in model.buffers:
                if buffer.grad_data is not None:
                    results.append(buffer.grad_data)
                if buffer.param_data is not None:
                    results.append(buffer.param_data)
    return results


def MegatronSwapManager(train_step_args, cmd_args):
    """
    Adapter to the megatron's train_step function.
    train_step_args is from the arguments of train_step.
    cmd_args is obtained from get_args() from megatron.
    """
    if len(train_step_args) < 4:
        raise ValueError("The length of arguments should be more than 4")
    model = train_step_args[2]
    optimizer = train_step_args[3]
    num_layers = cmd_args.num_layers
    return SwapManager(
        get_num_microbatches,
        model,
        num_layers,
        optimizer=optimizer,
        get_optimizer_tensors_fcn=megatron_get_optimizer_tensors_fcn,
        get_shared_tensors_fcn=megatron_get_shared_tensors_fcn,
    )


def train_step_wrapper(train_step):
    @wraps(train_step)
    def wrapper(*args, **kwargs):
        args_ = get_args()
        manager = MegatronSwapManager(args, args_)
        ret = train_step(*args, **kwargs)
        manager.step()
        return ret

    return wrapper
