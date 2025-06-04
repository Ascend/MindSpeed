# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import List

from .transformer_block import (
    transformer_block_backward
)
from .modules.utils import (
    LayerGraph
)



def gpt_model_backward(
    model_grad,
    layer_graphs: List[LayerGraph],
):
    block_input_grad = transformer_block_backward(model_grad, layer_graphs)

    return block_input_grad
