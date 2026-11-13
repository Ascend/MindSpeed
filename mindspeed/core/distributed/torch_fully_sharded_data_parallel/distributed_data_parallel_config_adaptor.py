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

import inspect
from functools import wraps

from mindspeed.args_utils import get_full_args


def distributed_data_parallel_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        known_config = {}
        unknown_config = {}
        full_args = vars(get_full_args()).copy()
        full_args.update(dict(kwargs))

        config_key = inspect.signature(self.__class__).parameters
        for key, value in full_args.items():
            if key in config_key:
                known_config[key] = value
            elif key == 'fsdp2_config_path':
                setattr(self, 'fsdp2_config_path', value)
            else:
                unknown_config[key] = value

        fn(self, *args, **known_config)

    return wrapper