#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import torch
from megatron.training.utils import get_args
from mindspeed.utils import set_actual_seq_len, set_position_ids, get_actual_seq_len, get_position_ids
from mindspeed.core.pipeline_parallel.fb_overlap.transformer_layer import transformer_layer_forward


def transformer_layer_recompute(
        bwd_layer_graph
):
    if not bwd_layer_graph.checkpointed:
        return bwd_layer_graph
    # Set actual_seq_len for recompute and avoid using actual_seq_len in forward context.
    # Also Record the forward context for restore.
    if get_args().reset_position_ids:
        fwd_actual_seq_len = get_actual_seq_len()
        fwd_position_ids = get_position_ids()
        set_actual_seq_len(bwd_layer_graph.actual_seq_len)
        set_position_ids(bwd_layer_graph.position_ids)

    # Recompute entire transformer layer for backward.
    with torch.enable_grad():
        _, _, restored_layer_graph = transformer_layer_forward(
            bwd_layer_graph.layer, bwd_layer_graph.layer_input, *bwd_layer_graph.layer_inputs, checkpoint=False
        )
        restored_layer_graph.unperm2_graph = (restored_layer_graph.unperm2_graph[0], bwd_layer_graph.unperm2_graph[1], restored_layer_graph.unperm2_graph[2])

    # Restore acutal_seq_len in forward context.
    if get_args().reset_position_ids:
        set_actual_seq_len(fwd_actual_seq_len)
        set_position_ids(fwd_position_ids)

    return restored_layer_graph
