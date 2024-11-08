import os
import time

from megatron.training import print_rank_0
from mindspeed.core.memory.adaptive_recomputing.swappable_tensor import SwappableTensor


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

    @staticmethod
    def is_allowed_wrap_tensor(tensor):
        if isinstance(tensor, SwappableTensor):
            return False
        # min wrap tensor size, default is 1024B
        config = os.getenv('MIN_SWAP_TENSOR_SIZE')
        min_swap_tensor_size = 1024
        if config is not None:
            try:
                min_swap_tensor_size = max(min_swap_tensor_size, int(config))
            except ValueError:
                print_rank_0('WARNING: MIN_SWAP_TENSOR_SIZE value error, fallback to default value 1024')
        if get_tensor_mem_size(tensor) < min_swap_tensor_size:
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
        if tensor.storage().size() != tensor.numel():
            wrapped_tensor.is_slice_tensor = True
        key = time.time()
        wrapped_tensor.set_tensor(key, tensor)
        self.device_tensors[key] = wrapped_tensor
        return wrapped_tensor

    def is_exist_tensor_allowed_swap(self):
        for tensor in self.device_tensors.values():
            if tensor.is_allowed_swap:
                return True
        return False
    
    def is_exist_tensor_contiguous(self):
        for tensor in self.device_tensors.values():
            if tensor.get_tensor().is_contiguous() and tensor.is_allowed_swap:
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
    
    def is_last_slice_shard_tensor_to_host(self, bro_key, bro_tensor):
        device_tensors_keys = list(self.device_tensors.keys())
        for key in device_tensors_keys:
            tensor = self.device_tensors[key]
            if key != bro_key and tensor.get_slice_tensor() and tensor.storage_data_ptr == bro_tensor.storage_data_ptr:
                return False
        return True

    def swap_out_by_size(self, size):
        """
        swap some tensors to host memory
        :param size: total size which is requested to release memory
        :return: true or false
        """
        print_rank_0("Need tensor size is : %d" % (size))
        if not self.device_tensors or not self.is_exist_tensor_allowed_swap():
            return False
        swap_size = 0
        swap_tensor_num = 0
        only_swap_contiguous_tensor = self.is_exist_tensor_contiguous()
        if only_swap_contiguous_tensor:
            cur_swap_size, cur_swap_tensor_num = self.traverse_swap_device_tensors(size, swap_size, False)
        else:
            cur_swap_size, cur_swap_tensor_num = self.traverse_swap_device_tensors(size, swap_size, True)
        swap_size += cur_swap_size
        swap_tensor_num += cur_swap_tensor_num
        if swap_size != 0:
            print_rank_0("swap tensor to CPU, tensor num: %d, release NPU memory size: %s (%d)" % (
                swap_tensor_num, hum_convert(swap_size), swap_size))
            print_rank_0("tensor nums wrap manager for [device: %d, CPU: %d]" % (
                len(self.device_tensors), len(self.host_tensors)))
        self.total_swap_out_size += swap_size
        return True
    
    def traverse_swap_device_tensors(self, size, swap_size, is_swap_not_contiguous):
        cur_swap_size = 0
        cur_swap_tensor_num = 0
        device_tensors_keys = list(self.device_tensors.keys())
        # swap device memory size multiple
        config = os.getenv('SWAP_SIZE_MULTIPLE')
        swap_size_multiple = 1
        if config is not None:
            try:
                swap_size_multiple = max(1, int(config))
            except ValueError:
                print_rank_0('WARNING: SWAP_SIZE_MULTIPLE value error, fallback to default value 1')
        for key in device_tensors_keys:
            if swap_size + cur_swap_size >= size * swap_size_multiple:
                break
            if key not in self.device_tensors.keys():
                continue
            tensor = self.device_tensors[key]
            if not is_swap_not_contiguous and not tensor.get_tensor().is_contiguous():
                continue
            if tensor.is_allowed_swap:
                tensor_size = 0
                if tensor.get_slice_tensor():
                    is_last_slice_tensor = self.is_last_slice_shard_tensor_to_host(key, tensor)
                    if is_last_slice_tensor:
                        tensor_size = tensor.get_tensor_origin_storage()
                        tensor.trans_to_cpu()
                    else:
                        tensor.slice_tensor_trans_to_cpu()
                else:
                    tensor_size = tensor.get_tensor().numel() * tensor.get_tensor().element_size()
                    tensor.trans_to_cpu()
                cur_swap_size += tensor_size
                self.device_tensors.pop(key)
                self.host_tensors[key] = tensor
                move_count = self.move_shard_tensor_to_host(key, tensor)
                cur_swap_tensor_num += 1 + move_count
        return cur_swap_size, cur_swap_tensor_num

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
                tensor.trans_to_device(False)
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
        cap_tensor.trans_to_device(True)
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