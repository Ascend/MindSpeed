# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from functools import wraps

import torch

from mindspeed.core.memory.swap_layer_input.swap_layer_input_manager import SwapLayerInputManager


def _get_hidden_states(args, kwargs):
    hidden_states = kwargs.get('hidden_states')
    if hidden_states is not None:
        return hidden_states
    if args and isinstance(args[0], torch.Tensor):
        return args[0]
    return None


def _get_backward_layer_graph(args, kwargs, name, position):
    layer_graph = kwargs.get(name)
    if layer_graph is not None and hasattr(layer_graph, 'layer') and hasattr(layer_graph, 'layer_input'):
        return layer_graph
    if len(args) > position:
        layer_graph = args[position]
        if layer_graph is not None and hasattr(layer_graph, 'layer') and hasattr(layer_graph, 'layer_input'):
            return layer_graph
    return None


def _swap_out_layer_input(layer, hidden_states):
    if (
        layer is None
        or getattr(layer, 'is_mtp', False)
        or not hasattr(layer, 'swap_manager')
        or not isinstance(hidden_states, torch.Tensor)
    ):
        return None

    hidden_states.swap_this_tensor = True
    return layer.swap_manager.swap_out_tensors([hidden_states])


def _restore_swap_entry(layer_graph):
    """Restore the exact input associated with a forward layer graph."""
    if layer_graph is None or not hasattr(layer_graph, 'layer') or not hasattr(layer_graph, 'layer_input'):
        return

    swap_entry = getattr(layer_graph, 'swap_layer_input_entry', None)
    manager = getattr(layer_graph.layer, 'swap_manager', None)
    if swap_entry is None or manager is None:
        return

    manager.restore_swap_entry(swap_entry)
    layer_graph.swap_layer_input_entry = None


def swap_layer_input_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        result = fn(self, *args, **kwargs)
        if not hasattr(self, 'swap_manager') and not getattr(self, 'is_mtp', False):
            self.swap_manager = SwapLayerInputManager(custom_check_fn=lambda x: getattr(x, 'swap_this_tensor', False))

        # MindSpeed-LLM MTP calls TransformerLayer twice, and the first call needs to be removed.
        if getattr(self, 'is_mtp', False):
            managers = SwapLayerInputManager.manager_map.get('default', [])
            if managers:
                managers.pop()

        return result

    return wrapper


def swap_layer_input_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if getattr(self, 'is_mtp', False) or not hasattr(self, 'swap_manager'):
            return fn(self, *args, **kwargs)

        hidden_states = _get_hidden_states(args, kwargs)
        if isinstance(hidden_states, torch.Tensor):
            hidden_states.swap_this_tensor = True

        if not torch.is_grad_enabled() and isinstance(hidden_states, torch.Tensor):
            self.swap_manager.swap_out_tensors([hidden_states])
            self.swap_manager.forward_hook()

        result = fn(self, *args, **kwargs)
        if isinstance(result, tuple) and result and isinstance(result[0], torch.Tensor) and result[0].requires_grad:
            result[0].register_hook(self.swap_manager.backward_hook)
        return result

    return wrapper


def swap_layer_input_fboverlap_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        swap_entry = _swap_out_layer_input(self, _get_hidden_states(args, kwargs))
        try:
            result = fn(self, *args, **kwargs)
        finally:
            if swap_entry is not None:
                self.swap_manager.wait_swap_out(swap_entry)

        if isinstance(result, tuple) and len(result) > 2 and swap_entry is not None:
            layer_graph = result[2]
            if layer_graph is not None and hasattr(layer_graph, 'layer') and hasattr(layer_graph, 'layer_input'):
                layer_graph.swap_layer_input_entry = swap_entry
        return result

    return wrapper


def swap_layer_input_fboverlap_1f1b_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        _restore_swap_entry(_get_backward_layer_graph(args, kwargs, 'bwd_layer_graph', 3))
        _restore_swap_entry(_get_backward_layer_graph(args, kwargs, 'next_bwd_layer_graph', 5))

        swap_entry = _swap_out_layer_input(self, _get_hidden_states(args, kwargs))
        try:
            result = fn(self, *args, **kwargs)
        finally:
            if swap_entry is not None:
                self.swap_manager.wait_swap_out(swap_entry)

        if isinstance(result, tuple) and len(result) > 2 and swap_entry is not None:
            layer_graph = result[2]
            if layer_graph is not None and hasattr(layer_graph, 'layer') and hasattr(layer_graph, 'layer_input'):
                layer_graph.swap_layer_input_entry = swap_entry
        return result

    return wrapper


def swap_layer_input_fboverlap_backward_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        layer_graph = args[1] if len(args) > 1 else kwargs.get('layer_graph')
        _restore_swap_entry(layer_graph)
        return fn(*args, **kwargs)

    return wrapper
