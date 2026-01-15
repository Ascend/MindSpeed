# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""
Tensor swapping system for efficient memory management in neural network training.

This module provides functionality to swap tensors between device and host memory
to optimize memory usage during forward and backward passes.
"""
from collections import defaultdict
from enum import Enum
from typing import Optional, Callable, List, Dict, Tuple

import torch


class TensorLocation(Enum):
    """Enumeration of possible tensor storage locations."""
    DEVICE = "device"
    HOST = "host"


def is_valid_for_swap(tensor: torch.Tensor, custom_check_fn: Optional[Callable] = None) -> bool:
    """
    Checks if a tensor is valid for swapping.

    Args:
        tensor: The tensor to validate.
        custom_check_fn: Optional custom validation function.

    Returns:
        bool: True if tensor can be swapped, False otherwise.
    """
    # Check if tensor is a parameter (should not be swapped)
    if (isinstance(tensor, torch.nn.parameter.Parameter) or
            isinstance(getattr(tensor, '_base', None), torch.nn.parameter.Parameter)):
        return False

    # Check if tensor storage is valid
    if tensor.storage().size() <= 0:
        return False

    # Apply custom validation if provided
    if custom_check_fn is not None and not custom_check_fn(tensor):
        return False

    return True


class TensorKeyManager:
    """Generates unique keys for tensor swap operations."""

    def __init__(self):
        self._current_layer_idx = -1
        # number of tensors to swap per layer
        self._layer_tensor_counts: Dict[int, int] = {}

    def get_key(self, layer_idx: int) -> Tuple[Tuple, bool]:
        """Generates a key for tensor swapping operations based on layer index and a flag
        indicating whether the previous layer has completed.

        This method computes a unique identifier (key) used to manage tensor swapping,
        typically in memory or state management systems. The key is composed of the
        current layer's index and an associated tensor index. It also returns a flag
        indicating whether the previous layer has completed its processing.

        Args:
            layer_idx (int): The index of the current layer for which the swap key
                is being generated.

        Returns:
            A tuple containing:
                - tensor_key (Tuple[int, int]): A tuple of (layer_idx, tensor_index),
                  serving as a unique key for tensor swap operations.
                - prev_layer_completed (bool): A boolean flag indicating whether
                  the processing of the previous layer (layer_idx - 1) is complete.
        """
        prev_layer_completed = False
        if layer_idx > self._current_layer_idx:
            self._layer_tensor_counts[layer_idx] = 1
            if layer_idx != 0:
                prev_layer_completed = True
        elif layer_idx == self._current_layer_idx:
            self._layer_tensor_counts[layer_idx] += 1
        else:
            # Step completed, reset for new iteration
            self._layer_tensor_counts = {layer_idx: 1}
        self._current_layer_idx = layer_idx

        tensor_index = self._layer_tensor_counts[self._current_layer_idx] - 1
        tensor_key = (self._current_layer_idx, tensor_index)

        return tensor_key, prev_layer_completed

    def get_prefetch_keys(self, layer_idx: int, tensor_idx: int) -> List[tuple]:
        """
        Get keys for tensors that should be prefetched.

        Args:
            layer_idx: Current layer index.
            tensor_idx: Current tensor index.

        Returns:
            List of prefetch keys.
        """
        prefetch_layer_idx = layer_idx - 1 if layer_idx >= 1 else None

        if prefetch_layer_idx is None:
            return []

        prefetch_layer_tensor_nums = self._layer_tensor_counts[prefetch_layer_idx]
        layer_tensor_nums = self._layer_tensor_counts[layer_idx]

        start_idx = tensor_idx * prefetch_layer_tensor_nums // layer_tensor_nums
        end_idx = (tensor_idx + 1) * prefetch_layer_tensor_nums // layer_tensor_nums

        prefetch_idx = range(start_idx, end_idx)
        return [(prefetch_layer_idx, prefetch_tensor_idx) for prefetch_tensor_idx in prefetch_idx]


class SwapTensor:
    """Represents a tensor that can be swapped between device and host memory."""

    def __init__(self, tensor: torch.Tensor, key: tuple):
        """
        Initialize swap tensor.

        Args:
            tensor: The original tensor to manage.
            key: Unique identifier for this swap tensor.
        """
        self.device_tensor = tensor
        self.size = tensor.size()
        self.storage_size = tensor.storage().size()
        self.host_tensor = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True, device='cpu')

        self.is_slice_tensor = tensor.storage().size() != tensor.numel()
        self.current_location = TensorLocation.DEVICE
        self.key = key

        self.h2d_event = torch.accelerator.Event()

    def async_d2h(self, stream: torch.Stream) -> None:
        """
        Asynchronously copy tensor from device to host.

        Args:
            stream: Stream to perform the operation on.
        """
        if self.current_location != TensorLocation.DEVICE:
            return

        forward_event = torch.accelerator.Event()
        forward_event.record()

        with torch.no_grad():
            with torch.accelerator.stream(stream):
                stream.wait_event(forward_event)
                if self.is_slice_tensor:
                    self.host_tensor.copy_(self.device_tensor, non_blocking=True)
                else:
                    self.host_tensor.storage().copy_(self.device_tensor.storage(), non_blocking=True)
                self.current_location = TensorLocation.HOST

    def wait_d2h_finished(self, stream: torch.Stream, should_wait_streams: bool) -> None:
        """
        Wait for device-to-host copy to complete.

        Args:
            stream: The stream used for copying.
            should_wait_streams: Whether to wait for streams to complete.
        """
        if self.current_location != TensorLocation.HOST:
            return

        if should_wait_streams:
            torch.accelerator.current_stream().wait_stream(stream)
            torch.accelerator.default_stream().wait_stream(stream)

        self.device_tensor.storage().resize_(0)
        self.current_location = TensorLocation.HOST

    def async_h2d(self, h2d_stream: torch.Stream,
                  should_resize_storage: bool, working_stream: Optional[torch.Stream] = None) -> None:
        """
        Asynchronously copy tensor from host to device.

        Args:
            h2d_stream: Stream for host-to-device transfer.
            should_resize_storage: Whether to resize device storage.
            working_stream: Optional working stream to synchronize with.
        """
        if self.current_location != TensorLocation.HOST:
            return

        backward_event = torch.accelerator.Event()
        backward_event.record()

        if should_resize_storage:
            self.device_tensor.storage().resize_(self.storage_size)

        with torch.no_grad():
            with torch.accelerator.stream(h2d_stream):
                h2d_stream.wait_event(backward_event)
                if self.is_slice_tensor:
                    self.device_tensor.copy_(self.host_tensor, non_blocking=True)
                else:
                    self.device_tensor.storage().copy_(self.host_tensor.storage(), non_blocking=True)
                self.h2d_event.record()
                self.current_location = TensorLocation.DEVICE

                if working_stream is not None:
                    working_stream.wait_stream(h2d_stream)
                else:
                    self.device_tensor.record_stream(h2d_stream)


class SingletonMeta(type):
    """
    single class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance

        return cls._instances[cls]


class AsyncSwapHandler(metaclass=SingletonMeta):
    """Central manager for coordinating tensor swap operations across the system."""

    def __init__(self, custom_check_fn: Optional[Callable] = None,):
        self.key_to_swap_tensor = {}
        self.tensor_key_to_ = {}
        self._key_manager = TensorKeyManager()
        self._d2h_stream: Optional[torch.Stream] = None
        self._h2d_stream: Optional[torch.Stream] = None
        self.custom_check_fn = custom_check_fn

    def set_custom_check_fn(self, fn):
        """Set a custom function to check tensor eligibility for swapping.

        Args:
            fn: The custom check function.
        """
        self.custom_check_fn = fn

    def swap_out_cur_and_release_prev(self, tensor: torch.Tensor, layer_idx: int, layer_nums: int):
        """async swap out current layer and release previous layer"""
        if not is_valid_for_swap(tensor, self.custom_check_fn):
            return tensor

        tensor_key, prev_layer_completed = self.get_swap_key(layer_idx)

        # release previous layer tensor
        if prev_layer_completed:
            self.release_device_tensor(layer_idx - 1)

        swap_tensor = SwapTensor(tensor, tensor_key)

        # The last layer does not need to swap because it will be used in backward soon.
        if layer_idx < layer_nums - 1:
            working_stream = torch.accelerator.current_stream()
            self.d2h_stream.wait_stream(working_stream)
            swap_tensor.async_d2h(self.d2h_stream)

        self.key_to_swap_tensor[tensor_key] = swap_tensor
        return swap_tensor

    def wait_cur_and_prefetch_next(self, tensor, prefetch=True):
        """wait current layer and prefetch next layer"""
        if not isinstance(tensor, SwapTensor):
            return tensor

        h2d_stream = self.h2d_stream
        d2h_stream = self.d2h_stream
        working_stream = torch.accelerator.current_stream()
        # make sure all d2h copy is done before into backward
        working_stream.wait_stream(h2d_stream)

        h2d_stream.wait_stream(working_stream)

        tensor.async_h2d(h2d_stream, True, working_stream)

        if prefetch:
            layer_idx, tensor_idx = tensor.key
            self.prefetch_tensors(layer_idx, tensor_idx, h2d_stream, d2h_stream)
        return tensor.device_tensor

    def get_swap_key(self, layer_idx):
        return self._key_manager.get_key(layer_idx)

    def exist(self, key):
        return key in self.key_to_swap_tensor

    def release_device_tensor(self, layer_idx):
        for tensor_key, swap_tensor in self.key_to_swap_tensor.items():
            if tensor_key[0] == layer_idx:
                swap_tensor.wait_d2h_finished(self.d2h_stream, True)

    def prefetch_tensors(self, layer_idx, tensor_idx, h2d_stream, d2h_stream):
        """Prefetch tensors to device memory.

        Args:
            layer_idx: Current layer index.
            tensor_idx: Current tensor index.
            h2d_stream: Stream for host-to-device transfers.
            d2h_stream: Stream for device-to-host transfers.
        """
        prefetch_keys = self._key_manager.get_prefetch_keys(layer_idx, tensor_idx)
        for prefetch_key in prefetch_keys:
            if self.exist(prefetch_key):
                swap_tensor = self.key_to_swap_tensor.pop(prefetch_key)
                d2h_stream.wait_stream(h2d_stream)
                swap_tensor.async_h2d(h2d_stream, True)
                swap_tensor.device_tensor.record_stream(h2d_stream)

    @property
    def d2h_stream(self) -> torch.Stream:
        """Get or create the device-to-host stream.

        Returns:
            The device-to-host stream.
        """
        if self._d2h_stream is None:
            self._d2h_stream = torch.accelerator.Stream()
        return self._d2h_stream

    @property
    def h2d_stream(self) -> torch.Stream:
        """Get or create the host-to-device stream.

        Returns:
            The host-to-device stream.
        """
        if self._h2d_stream is None:
            self._h2d_stream = torch.accelerator.Stream()
        return self._h2d_stream
    

class TensorSwapContext:
    """Context manager for tensor swap operations during model execution."""
    context_map = defaultdict(lambda: -1)

    def __init__(
            self,
            module_tag: str = 'default',
            custom_check_fn: Optional[Callable] = None,
            prefetch: bool = True) -> None:
        """Initialize the tensor swap context.

        Args:
            module_tag: A pattern string for identify structurally equivalent components across layers, enabling
                      consistent identification of the same sub-module role within different layers
                      The asterisk (`*`) acts as a wildcard, matching any layer index.

                      Examples:
                        1) For layers 'model.layer.0', 'model.layer.1', ... 'model.layer.N',
                           the module_tag would be 'model.layer.*'.

                        2) For the component 'up_proj' of expert 0 in different layers, such as
                           'model.layer.2.mlp.experts.0.up_proj' and 'model.layer.5.mlp.experts.0.up_proj',
                           the module_tag would be 'model.layer.*.mlp.experts.0.up_proj'.

                        3) For the component 'up_proj' of expert 1 in different layers, such as
                           'model.layer.3.mlp.experts.1.up_proj' and 'model.layer.7.mlp.experts.1.up_proj',
                           the module_tag would be 'model.layer.*.mlp.experts.1.up_proj'.

            custom_check_fn: Custom function to check tensor eligibility.
            prefetch: Whether to enable tensor prefetching.
        """
        self.module_tag = module_tag
        TensorSwapContext.context_map[self.module_tag] += 1
        self.layer_idx = TensorSwapContext.context_map[self.module_tag]
        self.prefetch = prefetch
        self.swap_handler = AsyncSwapHandler(custom_check_fn)

    def __enter__(self):
        """Enter the context and set up swap hooks."""
        def _on_save_for_backward(tensor):
            swap_tensor = self.swap_handler.swap_out_cur_and_release_prev(tensor, self.layer_idx, self.layer_nums)
            return swap_tensor

        def _on_get_saved_tensor(tensor) -> torch.Tensor:
            device_tensor = self.swap_handler.wait_cur_and_prefetch_next(tensor, self.prefetch)
            return device_tensor

        self.pack_hook = _on_save_for_backward
        self.unpack_hook = _on_get_saved_tensor
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.pack_hook, self.unpack_hook
        )

    def __exit__(self, *args):
        """Exit the context and remove hooks."""
        torch._C._autograd._pop_saved_tensors_default_hooks()

    def set_custom_check_fn(self, fn):
        """Set a custom function to check tensor eligibility for swapping.

        Args:
            fn: The custom check function.
        """
        self.swap_handler.set_custom_check_fn(fn)

    @property
    def layer_nums(self) -> int:
        """
        Returns:
            layer nums.
        """
        return TensorSwapContext.context_map[self.module_tag] + 1
