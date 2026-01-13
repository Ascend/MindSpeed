# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from typing import Callable, List, Optional  # codecheck_ignore
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps, partial
from enum import Enum

import torch
import torch_npu

from .utils import SimulationA2, ListNode, TensorState, TensorManager, ShareMemory


@dataclass
class GlobalContextConfig:
    """ Global layer activation context config

    Args:
        filter_funcs: Function for filtering non-target activation values.
        async_funcs: Names of operators that support asynchronous parallelism. (scalable)
    """
    moe_average_token_nums: int = 0
    num_experts: int = 0
    filter_funcs: List[Optional[Callable]] = field(default_factory=list)
    async_funcs: List[Optional[str]] = field(default_factory=list)


class ContextState(Enum):
    """ Define the current context state.
    """
    FORWARD = "forward"
    BACKWARD = "backward"


class GlobalContext:
    """ Global context for managing compressed tensors across layers in distributed training.
    
    Core Responsibilities:
    - Manages lifecycle of compressed activations across model layers.
    - Coordinates asynchronous compression/decompression during forward/backward passes.
    - Handles shared memory allocation for tensor storage.
    - Wraps critical operators (matmul/all2all/allgather) with compression logic.
    """
    def __init__(self, config: GlobalContextConfig) -> None:
        self.config = config
        self.simulation = SimulationA2()
        self.layer_activation_managers = OrderedDict()
        self.fwd_uuid_order = OrderedDict()
        self.bwd_uuid_order = OrderedDict()
        self.fwd_cur_layer_node = None
        self.bwd_cur_layer_node = None
        self.context_state = None
        self.vice_stream = torch.cuda.Stream()
        self.share_memorys = []
        self.absolute_order = 0
        self.lam_num = None
        self.statistic = True
        self.moe_average_token_nums = self.config.moe_average_token_nums
        self._wrapper_async_func()

    def get_absolute_order(self, is_first_step: bool) -> int:
        """ Provide the absolute order ID of the current computation 
        in the model forward pass.

        Args:
            is_first_step: Indicates whether the step is the first step.
        """
        if not is_first_step:
            if self.lam_num is None:
                self.lam_num = self.absolute_order
        if self.lam_num is not None:
            self.absolute_order = self.absolute_order % self.lam_num
        self.absolute_order += 1
        return self.absolute_order
        
    def filters(self, tensor) -> bool:
        """ Based on `filter_funcs`, determine whether the tensor meets the criteria.

        Args:
            tensor: Tensor to be filtered.
        """
        result = True
        for fn in self.config.filter_funcs:
            result &= fn(tensor)
        return result
    
    def push(self, order_layer_uuid, tensor) -> None:
        """ Store the tensor captured by the hook 
        in the layer management instance corresponding to the UUID.

        Args:
            order_layer_uuid: The UUID of the current layer.
            tensor: Tensor to be saved.
        """
        if self.filters(tensor):
            self.layer_activation_managers[order_layer_uuid].push(tensor)
    
    def pop(self, order_layer_uuid, tensor) -> None:
        """ Release the activation tensor stored at the corresponding layer based on the UUID.

        Args:
            order_layer_uuid: The UUID of the current layer.
            tensor: Tensor to be pop.
        """
        self.layer_activation_managers[order_layer_uuid].pop(tensor)

    def pack_start(self, order_layer_uuid) -> None:
        """ When entering the layer corresponding to the UUID, trigger the pre-operation.

        Args:
            order_layer_uuid: The UUID of the current layer.
        """
        self._order_forward(order_layer_uuid)
        self._create_layer_activation_manager(order_layer_uuid)
        self.context_state = ContextState.FORWARD

    def unpack_start(self, order_layer_uuid) -> None:
        """ When entering the layer backward pass 
        corresponding to the UUID, trigger the pre-operation.

        Args:
            order_layer_uuid: The UUID of the current layer.
        """
        self._order_backward(order_layer_uuid)
        self.layer_activation_managers[order_layer_uuid].pop_start()
        self.context_state = ContextState.BACKWARD

    def _order_forward(self, order_layer_uuid) -> None:
        """ Adjust the pointer of `fwd_cur_layer_node` to point to the layer corresponding to the UUID; 
        if it does not exist, create a layer node based on the relationship between the preceding and following nodes.

        Args:
            order_layer_uuid: The UUID of the current layer.
        """
        if order_layer_uuid in self.fwd_uuid_order:
            self.fwd_cur_layer_node = self.fwd_uuid_order[order_layer_uuid]
            return
        self.fwd_uuid_order[order_layer_uuid] = ListNode(order_layer_uuid, self.fwd_cur_layer_node)
        if self.fwd_cur_layer_node is not None:
            self.fwd_cur_layer_node.set_next_layer_node(self.fwd_uuid_order[order_layer_uuid])
        self.fwd_cur_layer_node = self.fwd_uuid_order[order_layer_uuid]

    def _order_backward(self, order_layer_uuid) -> None:
        """ Adjust the pointer of `bwd_cur_layer_node` to point to the layer corresponding to the UUID; 
        if it does not exist, create a layer node based on the relationship between the following and preceding nodes.

        Args:
            order_layer_uuid: The UUID of the current layer.
        """
        if order_layer_uuid in self.bwd_uuid_order:
            self.bwd_cur_layer_node = self.bwd_uuid_order[order_layer_uuid]
            return
        self.bwd_uuid_order[order_layer_uuid] = ListNode(order_layer_uuid, self.bwd_cur_layer_node)
        if self.bwd_cur_layer_node is not None:
            self.bwd_cur_layer_node.set_next_layer_node(self.bwd_uuid_order[order_layer_uuid])
        self.bwd_cur_layer_node = self.bwd_uuid_order[order_layer_uuid]

    def _create_layer_activation_manager(self, order_layer_uuid) -> None:
        """ Create corresponding activation management instances for each layer.

        Args:
            order_layer_uuid: The UUID of the current layer.
        """
        if order_layer_uuid in self.layer_activation_managers:
            self.layer_activation_managers[order_layer_uuid].push_start()
            return
        self.layer_activation_managers[order_layer_uuid] = LayerActivationManager(order_layer_uuid, self.vice_stream)
        self.layer_activation_managers[order_layer_uuid].push_start()
    
    def _pack(self, estimated_time, done=False) -> None:
        """ Start and end asynchronous compression tasks within the asynchronous operator wrapper.

        Args:
            estimated_time: The current estimated execution time of the asynchronous operator
            done: Indicate whether the asynchronous operator has completed. Defaults to False.
        """
        order_layer_uuid = self.fwd_cur_layer_node.order_layer_uuid
        need_pack_layer = self.fwd_uuid_order[order_layer_uuid].prev()
        if need_pack_layer is None:
            return
        need_pack_layer_uuid = need_pack_layer.order_layer_uuid
        lam = self.layer_activation_managers[need_pack_layer_uuid]
        if done:
            lam.pack_done()
            return
        plan_tensors = lam.plan(self.simulation.encode_max_numel(estimated_time))
        shares = OrderedDict()
        for pt in plan_tensors:
            shares[pt] = self._get_sm(pt)
        self.layer_activation_managers[need_pack_layer_uuid].pack(shares, self.statistic)

    def _unpack(self, estimated_time, done=False) -> None:
        """ Start and end asynchronous decompression tasks within the asynchronous operator wrapper.

        Args:
            estimated_time: The current estimated execution time of the asynchronous operator.
            done: Indicate whether the asynchronous operator has completed. Defaults to False.
        """
        order_layer_uuid = self.bwd_cur_layer_node.order_layer_uuid
        next_layer = self.bwd_uuid_order[order_layer_uuid].next()
        if next_layer is None:
            return
        need_unpack_layer_uuid = self.bwd_uuid_order[order_layer_uuid].next().order_layer_uuid
        lam = self.layer_activation_managers[need_unpack_layer_uuid]
        if done:
            lam.unpack_done()
            return
        plan_tensors = lam.plan(self.simulation.decode_max_numel(estimated_time), TensorState.COMPRESS)
        self.layer_activation_managers[need_unpack_layer_uuid].unpack(plan_tensors)

    def _apply(self, estimated_time, done=False) -> None:
        """ Dispatch compression or decompression tasks based on the current context state.

        Args:
            estimated_time: The current estimated execution time of the asynchronous operator.
            done: Indicate whether the asynchronous operator has completed. Defaults to False.
        """
        if self.context_state == ContextState.FORWARD:
            self._pack(estimated_time, done)
        elif self.context_state == ContextState.BACKWARD:
            self._unpack(estimated_time, done)
        else:
            raise ValueError(f"Unexpected context state: {self.context_state}")

    def _wrapper_async_func(self) -> None:
        """ Perform wrapper operations based on the given names of asynchronous operators.
        """
        for fn in self.config.async_funcs:
            if fn == "matmul":
                torch.matmul = self._wrapper_apply(torch.matmul, fn)
                torch.Tensor.matmul = self._wrapper_apply(torch.Tensor.matmul, fn)
            elif fn == "allgather":
                torch.distributed._all_gather_base = \
                    self._wrapper_apply(torch.distributed._all_gather_base, fn)
            elif fn == "all2all":
                torch.distributed.all_to_all_single = \
                    self._wrapper_apply(torch.distributed.all_to_all_single, fn)
            else:
                return

    def estimate_time(self, fn_name, *args, **kwargs) -> float:
        """ Estimate the execution duration based on the given asynchronous operator name and parameters.

        Args:
            fn_name: Asynchronous operator name.
        """
        return self.simulation.time_cost(fn_name, *args, **kwargs)

    def _wrapper_apply(self, func: Callable, fn_name: str) -> Callable:
        """ Apply lossless compression wrapper to all asynchronous functions.

        Args:
            func: Asynchronous function.
            fn_name: Asynchronous function name.
        """
        def wrapper(*args, **kwargs):
            if kwargs.get("async_op", False):
                return func(*args, **kwargs)
            self._apply(self.estimate_time(fn_name, *args, **kwargs))
            try:
                return func(*args, **kwargs)
            finally:
                self._apply(self.estimate_time(fn_name, *args, **kwargs), done=True)
        return wrapper

    def _get_sm(self, pt: torch.Tensor) -> ShareMemory:
        """ Obtain shared memory through the size and type of tensor.
        """
        dtype = pt.dtype
        numel = self._get_sm_target_numel(pt)
        for sm in self.share_memorys:
            if sm.can_be_used and numel == sm.numel and dtype == sm.dtype:
                sm.can_be_used = False
                return sm
        sm = ShareMemory(numel, dtype)
        sm.can_be_used = False
        self.share_memorys.append(sm)
        return sm

    def _get_sm_target_numel(self, pt: torch.Tensor) -> int:
        """
        Distinguish between `MoE` and `Dense` layers to reduce the overhead caused by memory allocation.
        """
        moe_special_activation_grad_fns = ["NpuSwigluBackward", "GMMFunctionBackward", "CppFunction"]
        if self.config.num_experts is not None:
            if self.config.num_experts > 0:
                for grad_fn in moe_special_activation_grad_fns:
                    if grad_fn in str(pt.grad_fn):
                        return pt.numel() // pt.shape[0] * self.moe_average_token_nums // pt.element_size()
        return pt.numel() // pt.element_size()


class LayerActivationManager:
    """Manages activation tensors' lifecycle with stream-aware compression for a specific layer.
    
    Handles tensor compression/decompression during forward/backward passes with:
    - Asynchronous stream operations (main stream vs compression stream)
    - Memory sharing via ShareMemory objects
    - State tracking for activation tensors
    
    Args:
        order_layer_uuid: Unique identifier for layer execution ordering
        vice_stream: Dedicated stream for compression operations
    """
    def __init__(self, order_layer_uuid, vice_stream: torch.cuda.Stream) -> None:
        self.order_layer_uuid = order_layer_uuid
        self.tensor_refs = {}
        self.vice_stream = vice_stream
        self.default_stream = torch.cuda.default_stream()
        self.pack_tensors = None
        self.unpack_tensors = None
    
    def push(self, tensor: torch.Tensor) -> None:
        """ The activation tensor captured by the hook.
        """
        if tensor in self.tensor_refs:
            return
        self.tensor_refs[tensor] = TensorManager(tensor)
    
    def pop(self, tensor: torch.Tensor) -> None:
        """ Release tensor reference.
        """
        if tensor in self.tensor_refs:
            del self.tensor_refs[tensor]

    def push_start(self) -> None:
        """ Check all tensors reference is None.
        """
        if len(self.tensor_refs) != 0:
            raise RuntimeError(f"Expected empty tensor_refs, but got {len(self.tensor_refs)}")

    def pop_start(self) -> None:
        """ Check for all tensors in current layer backward pass and recover it in time.
        """
        for ts in self.tensor_refs:
            if self.tensor_refs[ts].state != TensorState.NORMAL:
                self.tensor_refs[ts].recover()

    def pack(self, shares: OrderedDict[torch.Tensor, ShareMemory], statistic: bool) -> None:
        """ Allocate memory for the given tensor to be compressed 
        and asynchronously initiate the compression task.
        """
        self.pack_tensors = [ts for ts in shares]
        for ts in shares:
            self.tensor_refs[ts].malloc(shares[ts], statistic)
        self.vice_stream.wait_stream(self.default_stream)
        with torch.cuda.stream(self.vice_stream):
            for ts in shares:
                self.tensor_refs[ts].encode()
    
    def pack_done(self) -> None:
        """ Wait for the asynchronous compression task to complete 
        and release the original activation.
        """
        self.default_stream.wait_stream(self.vice_stream)
        for ts in self.pack_tensors:
            self.tensor_refs[ts].encode_wait()
        self.pack_tensors = None

    def unpack(self, choices: List[torch.Tensor]) -> None:
        """ Aasynchronously initiate the decompression task.
        """
        self.unpack_tensors = [ts for ts in choices]
        for ts in choices:
            self.tensor_refs[ts].pre_decode()
        self.vice_stream.wait_stream(self.default_stream)
        with torch.cuda.stream(self.vice_stream):
            for ts in choices:
                self.tensor_refs[ts].decode()

    def unpack_done(self) -> None:
        """ Wait for the asynchronous decompression task to complete 
        and recover the original activation.
        """
        self.default_stream.wait_stream(self.vice_stream)
        for ts in self.unpack_tensors:
            self.tensor_refs[ts].release()
        self.unpack_tensors = None

    def plan(self, max_numel: int, filter_state=TensorState.NORMAL) -> None:
        """ Select the tensor group to be compressed based on the given space constraints.
        """
        target_tensor = []
        target_numel = 0
        tensors = [ts for ts in self.tensor_refs if self.tensor_refs[ts].state == filter_state]
        for ts in tensors:
            target_numel += ts.numel()
            if target_numel >= max_numel:
                target_numel -= ts.numel()
                continue
            target_tensor.append(ts)
        return target_tensor


class CompressHook(torch.autograd.graph.saved_tensors_hooks):
    """ Hook for managing tensor compression during autograd computation.

    This hook implements pack/unpack logic to integrate with PyTorch's saved tensors
    mechanism, enabling custom compression during the backward pass.
    
    Args:
        order_layer_uuid: Unique identifier for the layer in execution order
        ctx: Global context for managing compressed tensors across layers
    """
    def __init__(self, order_layer_uuid, ctx: GlobalContext) -> None:
        self.global_ctx = ctx
        self.order_layer_uuid = order_layer_uuid
        self.output = None
        super().__init__(self.pack_hook, self.unpack_hook)

    def pack_hook(self, tensor: torch.Tensor) -> torch.Tensor:
        """ All tensors stored in the computational graph will trigger this hook.

        Args:
            tensor: The tensor required for backward pass.
        """
        self.global_ctx.push(self.order_layer_uuid, tensor)
        return tensor

    def unpack_hook(self, packed: torch.Tensor) -> torch.Tensor:
        """ All forward preserved activation tensors will trigger 
        this hook when used in backward computation.

        Args:
            packed: The tensor required for backward pass.
        """
        self.global_ctx.pop(self.order_layer_uuid, packed)
        return packed
