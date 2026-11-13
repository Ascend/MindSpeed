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

from abc import ABC, abstractmethod

from mindspeed.te.pytorch.fp8 import MatmulKey
from mindspeed.te.pytorch.module_typing import FP8Metadata


class CommOverlapConfig:

    @classmethod
    def get_tp_size(cls):
        if cls.tp_size is None:
            from megatron.core.parallel_state import get_tensor_model_parallel_world_size
            cls.tp_size = get_tensor_model_parallel_world_size()
        return cls.tp_size

    @classmethod
    def init_tp_size(cls, tp_size):
        cls.tp_size = tp_size

    @classmethod
    def get_tp_group(cls):
        if cls.tp_group is None:
            from megatron.core.parallel_state import get_tensor_model_parallel_group
            cls.tp_group = get_tensor_model_parallel_group()
        return cls.tp_group

    @classmethod
    def init_tp_group(cls, tp_group):
        cls.tp_group = tp_group

    tp_size = None
    tp_group = None
    save_allgather_input = True
    parallel_num = 2


class CommOverlapOps(ABC):

    @staticmethod
    @abstractmethod
    def allgather_matmul(input_, weight, bias, fp8_meta: FP8Metadata, key: MatmulKey):
        ...

    @staticmethod
    @abstractmethod
    def matmul_reduce_scatter(input_, weight, bias, fp8_meta: FP8Metadata, key: MatmulKey):
        ...

    @staticmethod
    @abstractmethod
    def matmul_all_reduce(input_, weight, bias, fp8_meta: FP8Metadata, key: MatmulKey, fp8_enable: bool):
        ...


COMM_OVERLAP_CONFIG = CommOverlapConfig()
