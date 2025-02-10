# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import os
from functools import wraps
from megatron.core import mpu
from .config.dist_train_config import get_dist_model_name


def get_checkpoint_name_wrapper(get_checkpoint_name):
    @wraps(get_checkpoint_name)
    def wrapper(*args, **kwargs):
        return _get_checkpoint_name(*args, **kwargs)
    return wrapper


def _get_checkpoint_name(checkpoints_path, iteration, release=False, **kwargs):
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)
    if kwargs.get('return_base_dir', False):
        common_path = os.path.join(checkpoints_path, directory)
        return common_path

    pipeline_parallel = (mpu.get_pipeline_model_parallel_world_size() > 1)
    tensor_rank = mpu.get_tensor_model_parallel_rank()
    pipeline_rank = mpu.get_pipeline_model_parallel_rank()
    model_name = get_dist_model_name()
    if not pipeline_parallel:
        common_path = os.path.join(checkpoints_path, directory,
                                   f'mp_{model_name}_rank_{tensor_rank:02d}')
    else:
        common_path = os.path.join(checkpoints_path, directory,
                                   f'mp_{model_name}_rank_{tensor_rank:02d}_{pipeline_rank:03d}')

    return os.path.join(common_path, "model_optim_rng.pt")
