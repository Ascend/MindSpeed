# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from typing import Optional, Callable, List

import torch
import torch_npu


def is_valid_for_swap(tensor: torch.Tensor, custom_check_fn: Optional[Callable] = None) -> bool:
    # Check if tensor is a parameter (should not be swapped)
    if isinstance(tensor, torch.nn.parameter.Parameter) or isinstance(
        getattr(tensor, '_base', None), torch.nn.parameter.Parameter
    ):
        return False

    # Check if tensor storage is valid
    if tensor.storage().size() <= 0:
        return False

    # Apply custom validation if provided
    if custom_check_fn is not None and not custom_check_fn(tensor):
        return False

    return True


class SwapTensors:
    """Manage a group of tensors' transmission between device and host.

    Each SwapTensors instance holds a List[Tensor] belonging to one
    micro-batch of one layer. It tracks per-tensor metadata (CPU buffers,
    storage sizes, slice flags) and a collective state for the group:
    "device" -> "d2h" -> "host" -> "h2d" -> "device".
    """

    def __init__(self, tensors):
        self.tensors: List[torch.Tensor] = tensors
        self.tensor_cpus: List[Optional[torch.Tensor]] = [
            torch.empty(t.shape, dtype=t.dtype, pin_memory=True, device='cpu') for t in self.tensors
        ]
        self.storage_sizes: List[int] = [t.storage().size() for t in self.tensors]
        self.is_slice_tensors: List[bool] = [t.storage().size() != t.numel() for t in self.tensors]
        self._stat = "device"

    def add_tensor(self, tensor: torch.Tensor):
        self.tensors.append(tensor)
        self.tensor_cpus.append(None)
        self.storage_sizes.append(tensor.storage().size())
        self.is_slice_tensors.append(tensor.storage().size() != tensor.numel())

    @property
    def stat(self):
        return self._stat

    def swap_to_host(self, stream, async_op=False):
        """Swap all tensors from device to host (D2H).

        Copies tensor data to pinned CPU memory and resizes device storage to 0
        to free device memory. Uses a separate NPU stream for async transfer.

        Args:
            stream: NPU stream for the D2H transfer.
            async_op: If True, the transfer is launched asynchronously.
                     If False, waits for completion before returning.
        """
        if self._stat != "device":
            return

        # Record current stream event to ensure compute is done before swap
        forward_event = torch.npu.Event()
        forward_event.record()

        with torch.no_grad():
            with torch_npu.npu.stream(stream):
                stream.wait_event(forward_event)
                for i, tensor in enumerate(self.tensors):
                    if tensor.storage().size() <= 0:
                        continue
                    if self.is_slice_tensors[i]:
                        self.tensor_cpus[i].copy_(tensor, non_blocking=True)
                    else:
                        self.tensor_cpus[i].storage().copy_(tensor.storage(), non_blocking=True)

        self._stat = "d2h"

        if not async_op:
            self.wait_d2h_and_release_device_tensor()

    def swap_to_device(self, stream, async_op=False):
        """Swap all tensors from host to device (H2D / prefetch).

        Resizes device storage back to original size and copies data from
        pinned CPU memory back to device. Uses a separate NPU stream for
        async transfer.

        Args:
            stream: NPU stream for the H2D transfer.
            async_op: If True, the transfer is launched asynchronously.
                     If False, waits for completion before returning.
        """
        if self._stat != "host":
            return

        # Record current stream event to ensure compute is done before prefetch
        backward_event = torch.npu.Event()
        backward_event.record()

        with torch.no_grad():
            with torch_npu.npu.stream(stream):
                stream.wait_event(backward_event)
                for i, tensor in enumerate(self.tensors):
                    if self.tensor_cpus[i] is None:
                        continue
                    tensor.storage().resize_(self.storage_sizes[i])
                    if self.is_slice_tensors[i]:
                        tensor.copy_(self.tensor_cpus[i], non_blocking=True)
                    else:
                        tensor.storage().copy_(self.tensor_cpus[i].storage(), non_blocking=True)

        self._stat = "h2d"

        if not async_op:
            self.wait_h2d_and_release_cpu_tensor()

    def wait_d2h_and_release_device_tensor(self):
        """Wait for D2H transfer to finish, then resize device storage to 0.

        After this call, tensors are fully on host memory and device memory
        is freed.
        """
        if self._stat != "d2h":
            return

        # Synchronize the d2h stream to ensure copies are complete
        stream = SwapLayerInputManager._d2h_stream
        if stream is not None:
            torch.npu.current_stream().wait_stream(stream)
            torch.npu.default_stream().wait_stream(stream)

        # Resize device storage to 0 to free device memory
        for i, tensor in enumerate(self.tensors):
            if self.tensor_cpus[i] is not None:
                tensor.storage().resize_(0)

        self._stat = "host"

    def wait_h2d_and_release_cpu_tensor(self):
        """Wait for H2D transfer to finish.

        After this call, tensors are fully back on device memory and ready
        for computation.
        """
        if self._stat != "h2d":
            return

        # Synchronize the h2d stream to ensure copies are complete
        stream = SwapLayerInputManager._h2d_stream
        if stream is not None:
            torch.npu.current_stream().wait_stream(stream)
            torch.npu.default_stream().wait_stream(stream)

        for i in range(len(self.tensor_cpus)):
            self.tensor_cpus[i] = None

        self._stat = "device"


class SwapLayerInputManager:
    """Context manager for tensor swap operations during model execution.

    Supports pipeline parallel (PP) by maintaining a batch_stack (FIFO deque)
    of SwapTensors, one per micro-batch. Each forward pass pushes a new
    SwapTensors entry onto the stack; swap-out (D2H) and prefetch (H2D)
    operate on the oldest entry (FIFO head) to respect the pipeline order.

    During the forward pass, after a layer completes, the previous layer's
    oldest batch tensors are swapped to host memory to free device memory.

    During the backward pass, before a layer needs its inputs, the previous
    layer's oldest batch tensors are prefetched from host to device.

    The last layer does not swap (no next layer needs to free its memory
    during forward, and no next layer needs to prefetch during backward).
    """

    manager_map = {}
    _h2d_stream = None
    _d2h_stream = None

    def __init__(
        self, module_tag: str = 'default', custom_check_fn: Optional[Callable] = None, prefetch: bool = True
    ) -> None:
        self.module_tag = module_tag
        self.custom_check_fn = custom_check_fn
        self.prefetch = prefetch
        self.batch_stack: List[SwapTensors] = []

        if self.module_tag not in SwapLayerInputManager.manager_map:
            SwapLayerInputManager.manager_map[self.module_tag] = []

        SwapLayerInputManager.manager_map[self.module_tag].append(self)
        self.layer_idx = SwapLayerInputManager.manager_map[self.module_tag].index(self)
        SwapLayerInputManager._ensure_streams()

    @classmethod
    def _ensure_streams(cls):
        """Lazily initialize the H2D and D2H NPU streams."""
        if cls._h2d_stream is None:
            cls._h2d_stream = torch_npu.npu.Stream(device=torch.npu.current_device())
        if cls._d2h_stream is None:
            cls._d2h_stream = torch_npu.npu.Stream(device=torch.npu.current_device())

    def swap_out_tensors(self, tensors: List[torch.Tensor]):
        """Add a tensor to the current micro-batch's swap collection."""
        if self.layer_idx + 1 < self.num_layers:
            swap_tensors = [t for t in tensors if is_valid_for_swap(t, self.custom_check_fn)]
            self.batch_stack.append(SwapTensors(swap_tensors))
            self.batch_stack[-1].swap_to_host(SwapLayerInputManager._d2h_stream, async_op=True)

    def wait_swap_out(self):
        if len(self.batch_stack) > 0:
            self.batch_stack[-1].wait_d2h_and_release_device_tensor()

    def swap_in_prev_layer(self):
        prev_manager = self._get_prev_layer_manager()
        if prev_manager is not None and len(prev_manager.batch_stack) > 0:
            for swap_entry in prev_manager.batch_stack:
                if swap_entry.stat == "host":
                    swap_entry.swap_to_device(SwapLayerInputManager._h2d_stream, async_op=True)
                    return

    def wait_swap_in(self):
        prev_manager = self._get_prev_layer_manager()
        if prev_manager is not None and len(prev_manager.batch_stack) > 0 and prev_manager.batch_stack[0].stat == 'h2d':
            prev_manager.batch_stack.pop(0).wait_h2d_and_release_cpu_tensor()

    def _get_prev_layer_manager(self) -> Optional['SwapLayerInputManager']:
        """Get the SwapLayerInputManager for the previous layer, or None if this is the first layer."""
        managers = SwapLayerInputManager.manager_map[self.module_tag]
        if self.layer_idx <= 0:
            return None
        return managers[self.layer_idx - 1]

    def forward_hook(self):
        self._ensure_streams()
        prev_manager = self._get_prev_layer_manager()
        if prev_manager is not None and len(prev_manager.batch_stack) > 0:
            swap_entry = prev_manager.batch_stack[-1]
            swap_entry.wait_d2h_and_release_device_tensor()

        if self.layer_idx + 1 < self.num_layers:
            self.batch_stack[-1].swap_to_host(SwapLayerInputManager._d2h_stream, async_op=True)

    def backward_hook(self, _):
        self._ensure_streams()
        if self.layer_idx + 1 < self.num_layers and self.batch_stack[0].stat == 'h2d':
            self.batch_stack.pop(0).wait_h2d_and_release_cpu_tensor()

        prev_manager = self._get_prev_layer_manager()
        if prev_manager is not None and len(prev_manager.batch_stack) > 0:
            swap_entry = prev_manager.batch_stack[0]
            swap_entry.swap_to_device(SwapLayerInputManager._h2d_stream, async_op=True)

    @property
    def num_layers(self) -> int:
        return len(SwapLayerInputManager.manager_map[self.module_tag])
