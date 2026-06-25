# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from functools import wraps

import torch

from mindspeed.core.memory.swap_layer_input.swap_layer_input_manager import SwapLayerInputManager
from mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules.utils import LayerGraph


def swap_layer_input_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        result = fn(self, *args, **kwargs)
        if not hasattr(self, 'swap_manager') and not getattr(self, 'is_mtp', False):
            self.swap_manager = SwapLayerInputManager(custom_check_fn=lambda x: getattr(x, 'swap_this_tensor', False))

        # MindSpeed-LLM MTP calls TransformerLayer twice, and the first call needs to be removed.
        if getattr(self, 'is_mtp', False):
            SwapLayerInputManager.manager_map["default"].pop(-1)

        return result

    return wrapper


def swap_layer_input_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'swap_manager'):
            return fn(self, *args, **kwargs)

        hidden_states = None
        if 'hidden_states' in kwargs:
            hidden_states = kwargs['hidden_states']
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = args[0]
        hidden_states.swap_this_tensor = True

        if not torch.is_grad_enabled():
            self.swap_manager.swap_out_tensors([hidden_states])
            self.swap_manager.forward_hook()
        result = fn(self, *args, **kwargs)
        if result[0].requires_grad:
            result[0].register_hook(self.swap_manager.backward_hook)
        return result

    return wrapper


def swap_layer_input_fboverlap_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'swap_manager'):
            return fn(self, *args, **kwargs)

        hidden_states = None
        if 'hidden_states' in kwargs:
            hidden_states = kwargs['hidden_states']
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = args[0]

        if kwargs.get('checkpoint', False):
            hidden_states.swap_this_tensor = True
            self.swap_manager.swap_out_tensors([hidden_states])
            result = fn(self, *args, **kwargs)
            self.swap_manager.wait_swap_out()
        else:
            result = fn(self, *args, **kwargs)

        return result

    return wrapper


def swap_layer_input_fboverlap_1f1b_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'swap_manager'):
            return fn(self, *args, **kwargs)

        hidden_states = None
        if 'hidden_states' in kwargs:
            hidden_states = kwargs['hidden_states']
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = args[0]

        backward_layer = None
        if 'bwd_layer_graph' in kwargs:
            backward_layer = kwargs['bwd_layer_graph'].layer
        elif len(args) > 3 and isinstance(args[3], LayerGraph):
            backward_layer = args[3].layer

        hidden_states.swap_this_tensor = True

        self.swap_manager.swap_out_tensors([hidden_states])
        backward_layer.swap_manager.swap_in_prev_layer()

        result = fn(self, *args, **kwargs)

        self.swap_manager.wait_swap_out()
        backward_layer.swap_manager.wait_swap_in()

        return result

    return wrapper


def swap_layer_input_fboverlap_backward_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        self = args[1].layer
        if not hasattr(self, 'swap_manager'):
            return fn(*args, **kwargs)

        self.swap_manager.swap_in_prev_layer()
        result = fn(*args, **kwargs)
        self.swap_manager.wait_swap_in()
        return result

    return wrapper
