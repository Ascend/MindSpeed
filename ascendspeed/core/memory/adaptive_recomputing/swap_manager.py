import os
import time
from copy import deepcopy

import torch
import torch_npu

from megatron import print_rank_0
from ascendspeed.core.memory.adaptive_recomputing.swappable_tensor import SwappableTensor


class SwapManagerMeta(type):
    swap_manager_instance = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.swap_manager_instance:
            instance = super().__call__(*args, **kwargs)
            cls.swap_manager_instance[cls] = instance
        return cls.swap_manager_instance[cls]        


class SwapManager(metaclass=SwapManagerMeta):
    def __init__(self):
        self.host_tensors = {}
        self.device_tensors = {}
        self.total_swap_out_size = 0
        self.origin_layers_peak_memory = {}
        self.policy_peak_memory = {}
        self.layers_interval_peak_memory = {}
        self.cur_pre_hook_layer_name = ""
        self.cur_post_hook_layer_name = ""
        self.swap_status = False

    @staticmethod
    def is_allowed_wrap_tensor(tensor):
        if isinstance(tensor, SwappableTensor):
            return False
        if not tensor.is_contiguous():
            return False
        # min wrap tensor size, default is 1024B
        config = os.getenv('MIN_SWAP_TENSOR_SIZE')
        min_swap_tensor_size = 1024
        if config is not None:
            min_swap_tensor_size = max(min_swap_tensor_size, int(config))
        if get_tensor_mem_size(tensor) < min_swap_tensor_size:
            return False
        # slice tensor
        if tensor.storage_offset() != 0 or tensor.storage().size() != tensor.numel():
            return False
        # leaf node tensor
        if tensor.grad_fn is None:
            return False

        return True

    def change_manager_tensor_status_to_allowed_swap(self):
        for k in self.device_tensors.keys():
            self.device_tensors[k].is_allowed_swap = True

    def wrap_tensor(self, tensor, pre_tensor_is_allowed_swap=False):
        """
        Wrap the original tensor.
        The tensor will be stored in the wrapped tensor. The original tensor may will be swap out to host cpu to release
        device memory when the swapping function is called
        :param pre_tensor_is_allowed_swap: pre tensor is allowed swap to CPU
        :param tensor: torch tensor which is needed to wrap
        :return: wrapped tensor
        """
        if pre_tensor_is_allowed_swap:
            self.change_manager_tensor_status_to_allowed_swap()
        if not self.is_allowed_wrap_tensor(tensor):
            return tensor
        wrapped_tensor = SwappableTensor(tensor)
        key = time.time()
        wrapped_tensor.set_tensor(key, tensor)
        self.device_tensors[key] = wrapped_tensor
        return wrapped_tensor

    def is_exist_tensor_allowed_swap(self):
        for tensor in self.device_tensors.values():
            if tensor.is_allowed_swap:
                return True
        return False

    def move_shard_tensor_to_host(self, bro_key, bro_tensor):
        move_count = 0
        device_tensors_keys = list(self.device_tensors.keys())
        for key in device_tensors_keys:
            tensor = self.device_tensors[key]
            if tensor.inner_tensor_data_ptr == bro_tensor.inner_tensor_data_ptr:
                self.device_tensors.pop(key)
                tensor.set_tensor_location("cpu")
                tensor.inner_tensor_bro_keys.append(bro_key)
                bro_tensor.inner_tensor_bro_keys.append(key)
                self.host_tensors[key] = tensor
                move_count += 1
        self.host_tensors[bro_key] = bro_tensor

        return move_count

    def swap_out_by_size(self, size, is_prediction=False):
        """
        swap some tensors to host memory
        :param size: total size which is requested to release memory
        :param is_prediction: is called by prediction
        :return: true or false
        """
        print_rank_0("Need tensor size is : %d" % (size))
        if not self.device_tensors or not self.is_exist_tensor_allowed_swap():
            return False
        layer_name = ""
        if not self.cur_pre_hook_layer_name and self.cur_pre_hook_layer_name in self.policy_peak_memory:
            layer_name = self.cur_pre_hook_layer_name
            size = max(size, self.policy_peak_memory[self.cur_pre_hook_layer_name])
        if not self.cur_post_hook_layer_name and self.cur_post_hook_layer_name in self.layers_interval_peak_memory:
            layer_name = self.cur_post_hook_layer_name
            size = max(size, self.layers_interval_peak_memory[self.cur_post_hook_layer_name])
        # swap device memory size multiple
        config = os.getenv('SWAP_SIZE_MULTIPLE')
        swap_size_multiple = 1
        if config is not None:
            swap_size_multiple = max(1, int(config))

        swap_size = 0
        swap_tensor_num = 0
        device_tensors_keys = list(self.device_tensors.keys())
        for key in device_tensors_keys:
            if swap_size >= size * swap_size_multiple:
                break
            if key not in self.device_tensors.keys():
                continue
            tensor = self.device_tensors[key]
            if tensor.is_allowed_swap:
                tensor_size = tensor.get_tensor().numel() * tensor.get_tensor().element_size()
                tensor.trans_to_cpu()
                swap_size += tensor_size
                self.device_tensors.pop(key)
                self.host_tensors[key] = tensor
                move_count = self.move_shard_tensor_to_host(key, tensor)
                swap_tensor_num += 1 + move_count
        if is_prediction and layer_name != "":
            print_rank_0(f"[peak prediction] layer name: {layer_name}")
        if swap_size != 0:
            print_rank_0("swap tensor to CPU, tensor num: %d, release NPU memory size: %s (%d)" % (
                swap_tensor_num, hum_convert(swap_size), swap_size))
            print_rank_0("tensor nums wrap manager for [device: %d, CPU: %d]" % (
                len(self.device_tensors), len(self.host_tensors)))
            self.swap_status = True
        self.total_swap_out_size += swap_size
        return True

    def unwrap_tensor(self, tensor):
        """
        Unwrap the tensor.
        If tensor is not on the device, the tensor will be swapped in to make sure that tensor is on device to compute.
        return the torch tensor to compute in torch graph
        :param tensor: wrapped tensor
        :return: origin tensor
        """
        if not isinstance(tensor, SwappableTensor):
            return tensor

        if tensor.id_key in self.host_tensors.keys():
            self.host_tensors.pop(tensor.id_key)
            if tensor.get_tensor().storage().size() == 0:
                self.move_shard_tensor_to_device(tensor)
        else:
            self.device_tensors.pop(tensor.id_key)

        return tensor.get_tensor()

    def move_shard_tensor_to_device(self, tensor):
        cap_tensor = tensor
        if tensor.inner_tensor_cpu_data is None:
            cap_key = tensor.inner_tensor_bro_keys[0]
            try:
                cap_tensor = self.host_tensors[cap_key]
            except KeyError:
                print_rank_0("[ERROR] The key doesn't exist.")
        cap_tensor.trans_to_device()
        if cap_tensor.id_key != tensor.id_key:
            cap_tensor.inner_tensor_bro_keys.remove(tensor.id_key)
            self.host_tensors.pop(cap_tensor.id_key)
            self.device_tensors[cap_tensor.id_key] = cap_tensor
        for key in cap_tensor.inner_tensor_bro_keys:
            bro_tensor = self.host_tensors.pop(key)
            bro_tensor.set_tensor_location("device")
            self.device_tensors[key] = bro_tensor

    def reset_swap_manager_tensors(self):
        self.device_tensors.clear()
        self.host_tensors.clear()
        self.cur_pre_hook_layer_name = ""
        self.cur_post_hook_layer_name = ""
        self.swap_status = False

    def reset_policy_peak_memory(self):
        self.policy_peak_memory = deepcopy(self.origin_layers_peak_memory)


def hum_convert(value):
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    origin_value = value
    for unit in units:
        if (value / 1024.0) < 1:
            return "%.2f%s" % (value, unit)
        value = value / 1024.0
    return "%.2f%s" % (origin_value, units[0])


def get_tensor_mem_size(tensor):
    return tensor.numel() * tensor.element_size()