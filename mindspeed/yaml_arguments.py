# Copyright 2024 Huawei Technologies Co., Ltd.
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
