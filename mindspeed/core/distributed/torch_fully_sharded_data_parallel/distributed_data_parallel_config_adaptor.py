# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

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
            elif key == 'fsdp2_config_str':
                setattr(self, 'fsdp2_config_str', value)
            else:
                unknown_config[key] = value

        fn(self, *args, **known_config)

    return wrapper