# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from mindspeed.core.tensor_parallel.comm_autograd_function import (
    auto_grad_scatter_along_first_dim_then_last_dim,
)
from mindspeed.core.tensor_parallel.comm_group_api import TPXCollectiveComm
from mindspeed.core.tensor_parallel.comm_group_api import TPYCollectiveComm
 
 
def language_model_embedding_forward_wrapper(forward):
    @wraps(forward)
    def wrapper(self, *args, **kwargs):
        encoder_input = forward(self, *args, **kwargs)
        if get_args().tp_2d:
            encoder_input = auto_grad_scatter_along_first_dim_then_last_dim(
                encoder_input, TPXCollectiveComm, TPYCollectiveComm
            )
        return encoder_input
    return wrapper
