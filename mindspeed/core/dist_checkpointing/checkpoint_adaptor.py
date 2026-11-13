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


def validate_global_plan_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return True
    return wrapper


def validate_non_overlapping_shards_metadata_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return
    return wrapper


def save_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        args_list = list(args)
        if len(args_list) > 5:
            args_list[4] = False
        if 'validate_access_integrity' in kwargs:
            kwargs['validate_access_integrity'] = False
        args = tuple(args_list)
        return func(*args, **kwargs)
    return wrapper