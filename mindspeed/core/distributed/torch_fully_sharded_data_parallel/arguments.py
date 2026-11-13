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


def _add_distributed_args_wrapper(_add_distributed_args):
    @wraps(_add_distributed_args)
    def wrapper(parser):
        parser = _add_distributed_args(parser)

         # Modify the distributed-backend argument to support extended configurations
        for action in parser._actions:
            if '--distributed-backend' in action.option_strings:
                action.choices = ['nccl', 'gloo', 'npu:hccl,cpu:gloo']
                action.help = 'Backend for distributed training. Single: "nccl" or "gloo", Multiple: "nccl,gloo"'
                break
        return parser
    return wrapper