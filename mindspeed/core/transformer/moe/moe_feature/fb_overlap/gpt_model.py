# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd.
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

from typing import List
from torch import Tensor

from .transformer_block import (
    transformer_block_backward
)
from .modules.utils import (
    LayerGraph
)



def gpt_model_backward(
    model_grad,
    layer_graphs: List[LayerGraph]
):
    block_input_grad = transformer_block_backward(model_grad, layer_graphs)

    return block_input_grad
