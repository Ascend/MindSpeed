#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import Optional, Dict, List

import torch
from torch.autograd.variable import Variable
from megatron.core.pipeline_parallel import p2p_communication


def detach_tensor(tensor, checkpoint_forward=False):
    if checkpoint_forward:
        return tensor
    if tensor is None:
        return None
    detached_tensor = tensor.detach()
    detached_tensor.requires_grad = True
    return detached_tensor


def run_graph_backward(graph, output_tensor_grad=None, keep_graph=False, keep_grad=False):
    if isinstance(graph[0], torch.Tensor):
        grad_tensors = [output_tensor_grad]
        if output_tensor_grad is None and graph[1] is not None and graph[1].grad is not None:
            grad_tensors = [graph[1].grad]
        tensors = [graph[0]]
    else:
        grad_tensors = filter(lambda x: x is not None, output_tensor_grad) if output_tensor_grad is not None else None
        if output_tensor_grad is None and graph[1] is not None and graph[1][0].grad is not None:
            grad_tensors = [x.grad for x in filter(lambda x:x is not None, graph[1])]
        tensors = filter(lambda x: x is not None, graph[0])
    Variable._execution_engine.run_backward(
        tensors=(*tensors,),
        grad_tensors=(*grad_tensors,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )

    if not keep_graph:
        for tensor in tensors:
            tensor.untyped_storage().resize_(0)
    if not keep_grad:
        for grad in grad_tensors:
            grad.untyped_storage().resize_(0)



class NoopLayerGraph:
    def __init__(self, layer_input, layer_output, layer, checkpointed=False):
        self.layer_input = layer_input
        if not checkpointed:
            self.unperm2_graph = (layer_output, None)
        else:
            self.unperm2_graph = (None, None)
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
            self.unperm2_graph = (None, None)

        self.layer_input = saved_graph_and_graph_inputs[-1]
        self.recompute_needed_tensors = recompute_needed_tensors
        self.input_splits = input_splits
        self.output_splits = output_splits
        self.checkpointed = checkpointed
        self.layer = layer
        self.is_moe_layer = hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts')


    def record_layer_inputs(self, *args):
        self.layer_inputs = args


class P2PCommParams:
    tensor_shape = None
    config = None

    def __init__(self, send_next=False, send_prev=False, recv_next=False, recv_prev=False):
        self.send_next = send_next
        self.send_prev = send_prev
        self.recv_next = recv_next
        self.recv_prev = recv_prev

    def __str__(self):
        return f'send next:{self.send_next} send_prev:{self.send_prev} recv_next:{self.recv_next} recv_prev:{self.recv_prev}'


class P2PCommOutput:
    def __init__(self, input_tensor=None, output_tensor_grad=None, fwd_wait_handles=None, bwd_wait_handles=None, input_tensor_grad=None):
        self.input_tensor = input_tensor
        self.fwd_wait_handles = fwd_wait_handles
        self.output_tensor_grad = output_tensor_grad
        self.bwd_wait_handles = bwd_wait_handles
        self.input_tensor_grad = input_tensor_grad


def is_p2p_comm_needed(pp_comm_params: P2PCommParams):
    return pp_comm_params is not None and \
           (pp_comm_params.send_next or pp_comm_params.send_prev or pp_comm_params.recv_next or pp_comm_params.recv_prev)


def p2p_comm_helper(comm_params: P2PCommParams, tensor_tosend):
    assert not (comm_params.send_next and comm_params.send_prev)
    assert not (comm_params.recv_next and comm_params.recv_prev)
    tensor_send_next = None
    if comm_params.send_next:
        tensor_send_next = tensor_tosend
    tensor_send_prev = None
    if comm_params.send_prev:
        tensor_send_prev = tensor_tosend
    tensor_recv_prev, tensor_recv_next, p2p_handles = p2p_communication._communicate(
        tensor_send_next=tensor_send_next,
        tensor_send_prev=tensor_send_prev,
        recv_prev=comm_params.recv_prev,
        recv_next=comm_params.recv_next,
        tensor_shape=comm_params.tensor_shape,
        wait_on_reqs=False,
        config=comm_params.config
    )

    if comm_params.recv_next:
        return tensor_recv_next, p2p_handles
    elif comm_params.recv_prev:
        return tensor_recv_prev, p2p_handles
    else:
        return None, p2p_handles


class TensorSwapManager:
    """Manager for asynchronous tensor swapping between NPU and CPU memory."""
    
    _SWAP_OUT_STREAM = None
    _SWAP_IN_STREAM = None
    _ALL_SWAP_OUT_QUEUES = {}  # Dictionary of swap groups for coordinated operations

    @classmethod
    def _get_swap_out_stream(cls):
        """Get or create the swap-out stream."""
        if cls._SWAP_OUT_STREAM is None:
            cls._SWAP_OUT_STREAM = torch.npu.Stream(device=torch.npu.current_device())
        return cls._SWAP_OUT_STREAM

    @classmethod
    def _get_swap_in_stream(cls):
        """Get or create the swap-in stream."""
        if cls._SWAP_IN_STREAM is None:
            cls._SWAP_IN_STREAM = torch.npu.Stream(device=torch.npu.current_device())
        return cls._SWAP_IN_STREAM

    def __init__(self, tensor, swap_group_name=None):
        """
        Initialize a tensor swap manager.
        
        Args:
            tensor: The NPU tensor to swap
            swap_group_name: Optional group name for coordinated swap operations
        """
        # Check if tensor is a slice (requires special handling, not supported yet)
        if tensor.storage().size() != tensor.numel():
            raise AssertionError('TensorSwapManager cannot handle sliced tensors')

        self.npu_tensor = tensor
        self.cpu_tensor = None
        self.swap_out_event = None
        self.swap_in_event = None

        self.swap_group_name = swap_group_name  # Group for coordinated operations
        self.under_swap_in = False

        # Initialize queue for this swap group if it doesn't exist
        if swap_group_name and swap_group_name not in self._ALL_SWAP_OUT_QUEUES:
            self._ALL_SWAP_OUT_QUEUES[swap_group_name] = []

    def async_swap_out(self, wait_event=None, wait_stream=None):
        """
        Asynchronously copy tensor from NPU to CPU memory.
        
        Args:
            event: Optional event to wait for before swapping
            stream: Optional stream to synchronize with before swapping
        """
        swap_stream = self._get_swap_out_stream()
        # Allocate pinned CPU memory (enables faster async transfers)
        self.cpu_tensor = torch.empty_like(self.npu_tensor, 
                                         pin_memory=True, 
                                         device='cpu')
        
        with torch.npu.stream(swap_stream):
            if wait_event:
                swap_stream.wait_event(wait_event)
            if wait_stream:
                swap_stream.wait_stream(wait_stream)

            self.cpu_tensor.untyped_storage().copy_(
                self.npu_tensor.untyped_storage(), 
                non_blocking=True)
            
            self.swap_out_event = torch.npu.Event()
            self.swap_out_event.record()
            
            # Add to swap group if specified
            if self.swap_group_name:
                self._ALL_SWAP_OUT_QUEUES[self.swap_group_name].append(self)

    def wait_swap_out(self):
        """Wait for swap-out to complete and release NPU memory."""
        if self.swap_out_event:
            torch.npu.current_stream().wait_event(self.swap_out_event)
        # Release NPU memory (but keep storage object alive)
        if not self.under_swap_in:
            self.npu_tensor.untyped_storage().resize_(0)

    def async_swap_in(self, wait_event=None, wait_stream=None):
        """
        Asynchronously copy tensor from CPU back to NPU memory.
        
        Args:
            event: Optional event to wait for before swapping
            stream: Optional stream to synchronize with before swapping
        """
        self.under_swap_in = True
        if self.npu_tensor.untyped_storage().size() != 0:
            return
        swap_stream = self._get_swap_in_stream()
        # Ensure NPU storage is properly sized
        self.npu_tensor.untyped_storage().resize_(
            self.cpu_tensor.untyped_storage().size())
        # Wait for previous swap-out to complete
        torch.npu.current_stream().wait_event(self.swap_out_event)
        
        with torch.npu.stream(swap_stream):
            if wait_event:
                swap_stream.wait_event(wait_event)
            if wait_stream:
                swap_stream.wait_stream(wait_stream)

            self.npu_tensor.untyped_storage().copy_(
                self.cpu_tensor.untyped_storage(), 
                non_blocking=True)
            
            self.swap_in_event = torch.npu.Event()
            self.swap_in_event.record()

    def wait_swap_in(self):
        """Wait for swap-in to complete and release CPU memory."""
        if self.swap_in_event:
            torch.npu.current_stream().wait_event(self.swap_in_event)
        # Release CPU memory
        self.cpu_tensor = None
        self.under_swap_in = False

    @classmethod
    def wait_all_swap_out(cls, group_name):
        """
        Wait for all swap-out operations in a group to complete.
        
        Args:
            group_name: Name of the swap group to synchronize
        """
        if group_name in cls._ALL_SWAP_OUT_QUEUES:
            for manager in cls._ALL_SWAP_OUT_QUEUES[group_name]:
                manager.wait_swap_out()
            # Clear the group while maintaining the list object
            cls._ALL_SWAP_OUT_QUEUES[group_name].clear()


def make_wait_swap_in_hook(swap_manager):
    """
    Create a hook that waits for a swap-in operation to complete.
    
    Returns:
        A callable hook function that waits for swap-in completion
    """
    return lambda *_: swap_manager.wait_swap_in()


def make_async_swap_in_hook(swap_managers):
    """
    Create a hook that initiates async swap-in for multiple tensors.
    
    Args:
        swap_managers: List of TensorSwapManager instances
        
    Returns:
        A callable hook function that triggers swap-in for all managers
    """
    return lambda *_: [m.async_swap_in(wait_stream=torch.npu.current_stream()) 
                      for m in swap_managers]
