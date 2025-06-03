#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import torch
import mindspore
from torch.autograd import recompute_instance
from mindspore.common.api import _convert_python_data


def detach_tensor(tensor, checkpoint_forward=False):
    if checkpoint_forward:
        return tensor
    if tensor is None:
        return None
    detached_tensor = mindspore.ops.stop_gradient(tensor)
    detached_tensor.requires_grad = True
    return detached_tensor


def run_graph_backward(graph, output_tensor_grad=None, keep_graph=False, keep_grad=False):
    if output_tensor_grad is None:
        output_tensor_grad = []
    elif isinstance(output_tensor_grad, torch.Tensor):
        output_tensor_grad = [output_tensor_grad]
    output_tensor_grad = list(filter(lambda x: x is not None, output_tensor_grad))

    grad_tensors = output_tensor_grad
    vjp_func = graph[2]
    if grad_tensors is None or len(grad_tensors) == 0:
        inputs_grad = (None,)
    else:
        inputs_grad = _convert_python_data(vjp_func(*grad_tensors))
    vjp_func = None

    return inputs_grad


def dummy_forward_step_func():
    '''
        dummy function for calling _pynative_executor.new_graph/end_graph
    '''
    pass


def run_graph_forward(func, *inputs):
    if not recompute_instance.recompute:
        # with torch.enable_grad():
        output, f_vjp = torch.autograd.vjp(func, *inputs)
    else:
        output = func(*inputs)
        f_vjp = None

    return output, f_vjp


class NoopLayerGraph:
    def __init__(self, layer_input, layer_output, layer, checkpointed=False):
        self.layer_input = layer_input
        if not checkpointed:
            self.unperm2_graph = (layer_output, None, None)
        else:
            self.unperm2_graph = (None, None, None)
        self.checkpointed = checkpointed
        self.layer = layer

    def record_layer_inputs(self, *args):
        self.layer_inputs = args


class LayerGraph:
    def __init__(self, saved_graph_and_graph_inputs, recompute_needed_tensors, input_splits, output_splits, layer, checkpointed=False):
        if not checkpointed:
            self.attn_graph = saved_graph_and_graph_inputs[0]
            self.pre_mlp_layernorm_graph = saved_graph_and_graph_inputs[1]
            self.router_graph = saved_graph_and_graph_inputs[2]
            self.perm1_graph = saved_graph_and_graph_inputs[3]
            self.perm_a2a_graph = saved_graph_and_graph_inputs[4]
            self.perm2_graph = saved_graph_and_graph_inputs[5]
            self.grouped_mlp_graph = saved_graph_and_graph_inputs[6]
            self.unperm1_graph = saved_graph_and_graph_inputs[7]
            self.unperm_a2a_graph = saved_graph_and_graph_inputs[8]
            self.unperm2_graph = saved_graph_and_graph_inputs[9]
            self.shared_experts_graph = saved_graph_and_graph_inputs[10]
        else:
            self.unperm2_graph = (None, None, None)

        self.layer_input = saved_graph_and_graph_inputs[-1]
        self.recompute_needed_tensors = recompute_needed_tensors
        self.input_splits = input_splits
        self.output_splits = output_splits
        self.checkpointed = checkpointed
        self.layer = layer
        self.is_moe_layer = hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts')
        # For Swap attention activation
        self.attn_swap_managers = None
        self.unperm2_swap_manager = None


    def record_layer_inputs(self, *args):
        self.layer_inputs = args
