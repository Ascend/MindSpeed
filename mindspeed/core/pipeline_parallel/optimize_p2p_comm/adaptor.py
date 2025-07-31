
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.


from functools import wraps


def core_transformer_config_from_args_wrapper(fn):
    @wraps(fn)
    def wrapper(args):
        config = fn(args)
        config.batch_p2p_comm = False
        return config

    return wrapper

