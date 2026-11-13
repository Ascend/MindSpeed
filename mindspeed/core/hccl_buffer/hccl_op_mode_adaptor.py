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

import torch_npu

from megatron.training.utils import print_rank_0

from mindspeed.args_utils import get_full_args as get_args
from mindspeed.core.hccl_buffer.hccl_adaptive_func import parse_hccl_op_mode_string, _HCCL_OP_MODE


def get_nccl_options_wrapper(get_nccl_options):
    @wraps(get_nccl_options)
    def wrapper(pg_name, nccl_comm_cfgs):
        args = get_args()
        options = get_nccl_options(pg_name, nccl_comm_cfgs)
        if args.hccl_op_mode and _HCCL_OP_MODE.get(pg_name) is not None:
            original_hccl_config = options.hccl_config
            options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
            original_hccl_config["hccl_op_expansion_mode"] = _HCCL_OP_MODE[pg_name]
            options.hccl_config = original_hccl_config
        return options

    return wrapper


def hccl_op_mode_set_wrapper(initialize_model_parallel):
    @wraps(initialize_model_parallel)
    def wrapper(*args, **kwargs):
        config = get_args()
        if config.hccl_op_mode is not None:
            parse_hccl_op_mode_string(config.hccl_op_mode)
            print_rank_0(f"hccl_op_mode_set: {_HCCL_OP_MODE}")

        return initialize_model_parallel(*args, **kwargs)

    return wrapper
