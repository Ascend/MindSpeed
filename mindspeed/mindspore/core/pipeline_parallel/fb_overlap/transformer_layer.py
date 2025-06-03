#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from contextlib import nullcontext
import torch
from mindspeed.core.transformer.transformer_block import NoopTransformerLayer

from mindspeed.core.pipeline_parallel.fb_overlap.transformer_layer import transformer_layer_forward

from mindspeed.core.pipeline_parallel.fb_overlap.modules.utils import NoopLayerGraph

from mindspeed.mindspore.core.pipeline_parallel.fb_overlap.overlap_funcs import (
    transformer_layer_backward_moe,
    transformer_layer_backward_dense,
    transformer_layer_backward_noop,
)



def transformer_layer_backward(
    layer_output_grad,
    layer_graph
):
    if layer_graph.checkpointed:
        with torch.enable_grad():
            _, _, restored_layer_graph = transformer_layer_forward(
                layer_graph.layer, layer_graph.layer_input, *layer_graph.layer_inputs, checkpoint=False
            )
            restored_layer_graph.unperm2_graph = (restored_layer_graph.unperm2_graph[0], layer_graph.unperm2_graph[1], restored_layer_graph.unperm2_graph[2])
            layer_graph = restored_layer_graph
    if isinstance(layer_graph, NoopLayerGraph):
        return transformer_layer_backward_noop(layer_output_grad, layer_graph)
    elif layer_graph.is_moe_layer:
        return transformer_layer_backward_moe(layer_output_grad, layer_graph)
    else:
        return transformer_layer_backward_dense(layer_output_grad, layer_graph)
