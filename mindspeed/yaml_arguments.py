# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

from functools import wraps


def core_transformer_config_from_yaml_wrapper(fn):
    @wraps(fn)
    def wrapper(args, transfomer_key):
        config = fn(args, "language_model")
        config.context_parallel_algo = args.context_parallel_algo
        config.batch_p2p_comm = False
        if args.use_multiparameter_pipeline_model_parallel:
            config.deallocate_pipeline_outputs = False
        return config

    return wrapper


def print_args_wrapper(fn):
    @wraps(fn)
    def wrapper(title, args, after_validate=False):
        if after_validate:
            fn(title, args)

    return wrapper
