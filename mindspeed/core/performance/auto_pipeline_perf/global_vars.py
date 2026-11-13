# Copyright 2026 Huawei Technologies Co., Ltd.
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
from megatron.training import get_args


def get_num_microbatches_wrapper(get_num_microbatches):
    @wraps(get_num_microbatches)
    def wrapper(*args, **kwargs):
        argument = get_args()
        automated_pipeline_profile = argument.automated_pipeline_perf and not argument.optimized_mbs_list
        if argument.automated_pipeline_perf and argument.optimized_mbs_list and argument.optimized_mbs_mode:
            return len(argument.optimized_mbs_list)
        elif automated_pipeline_profile:
            return argument.global_batch_size // argument.data_parallel_size // argument.micro_batch_size
        else:
            return get_num_microbatches(*args, **kwargs)
    return wrapper
