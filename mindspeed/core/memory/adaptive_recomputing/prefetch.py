# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import re
import torch
import torch_npu
from megatron.training import get_args


def get_layer_id(name):
    if name:
        matches = re.findall(r'\.(\d+)\.?', str(name))
        if matches:
            return matches[0]
        return -1
    return -1


class SwapTensor:
    def __init__(self, tensor, layer_name):
        self.tensor = tensor
        self.size = tensor.size()
        self.storage_size = tensor.storage().size()
        self.tensor_cpu = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True, device='cpu')

        self.d2h_event = None
        self.h2d_event = torch.npu.Event()

        self.stat = "device"
        self.layer_name = layer_name

        self.prefetch_data_ptr = tensor.data_ptr()
        self.storage_data_ptr = tensor.storage().data_ptr()
        self.layer_id = None
        self.first_tensor = False
        self.last_tensor = False
        self.is_slice_tensor = tensor.storage().size() != tensor.numel()
        self.stream = None
        self.layer_index = 0

    # device to host
    def launch_d2h(self, stream):
        if self.stat != "device":
            return
        forward_event = torch.npu.Event()
        forward_event.record()
        with torch.no_grad():
            with torch_npu.npu.stream(stream):
                stream.wait_event(forward_event)
                if self.is_slice_tensor:
                    self.tensor_cpu.copy_(self.tensor, non_blocking=True)
                else:
                    self.tensor_cpu.storage().copy_(self.tensor.storage(), non_blocking=True)
                self.stat = "d2h"

    # synchronize d2h and resize 0
    def wait_d2h_finished(self, stream, need_wait=False):
        if self.stat != "d2h":
            return
        if need_wait:
            torch.npu.current_stream().wait_stream(stream)
            torch.npu.default_stream().wait_stream(stream)
        self.tensor.storage().resize_(0)
        self.stat = "host"

    # resize storage_size and host to device
    def launch_h2d(self, stream, flag):
        if self.stat != "host":
            return
        backward_event = torch.npu.Event()
        backward_event.record()
        if flag:
            self.tensor.storage().resize_(self.storage_size)
        with torch.no_grad():
            with torch_npu.npu.stream(stream):
                stream.wait_event(backward_event)
                if self.is_slice_tensor:
                    self.tensor.copy_(self.tensor_cpu, non_blocking=True)
                else:
                    self.tensor.storage().copy_(self.tensor_cpu.storage(), non_blocking=True)
                self.h2d_event.record()
                self.stat = "h2d"

    # synchronize h2d
    def wait_h2d_finished(self, stream, need_wait=False):
        if self.stat != "h2d":
            return
        if need_wait:
            torch.npu.current_stream().wait_stream(stream)
            torch.npu.default_stream().wait_stream(stream)
        self.stat = "device"


class SwapPrefetch:
    swap_prefetch = None

    def __init__(self, prefetch_args):
        swap_list, vpp, interval, num_prefetch = prefetch_args
        all_args = get_args()
        self.prefetch_stream = torch_npu.npu.Stream(device=torch.npu.current_device())
        self.pp = all_args.pipeline_model_parallel_size
        self.vpp = min(vpp, num_prefetch)
        self.first_layer_id = 0
        if isinstance(all_args.noop_layers, set):
            for layer_id in swap_list[0]:
                if layer_id != '':
                    self.first_layer_id = int(layer_id)
                    break

        self.swap_tensors = []
        self.layer_name = ""

        self.data_ptr = {}
        self.prefetch_list = []
        self.prefetch_data_ptr_list = []
        self.cur_micro_num = 0
        self.remove_num = 0
        self.forward_flag = False
        self.interval = interval
        self.slice_tensor_storage_ptr = {}
        self.slice_tensor_storage_ptr_list = []
        self.eval_end_flag = False

    @staticmethod
    def no_swap_tensor(ori_tensor):
        if ori_tensor.numel() * ori_tensor.element_size() * 2 < 1024 * 1024:
            return True
        if ori_tensor.grad_fn is None:
            return True
        if ori_tensor.storage().size() == 0:
            return True
        if ori_tensor.storage().size() != ori_tensor.numel():
            return True
        if ori_tensor._base is not None and ori_tensor._base.dim() >= 5:
            return True

        return False

    def pack_hook(self, ori_tensor):
        args = get_args()
        if args.eval_interval:
            if args.curr_iteration % args.eval_interval != 0:
                self.eval_end_flag = False
            if args.curr_iteration and args.curr_iteration % args.eval_interval == 0 and not self.eval_end_flag:
                self.prefetch_data_ptr_list = []
                self.prefetch_list = []
                self.slice_tensor_storage_ptr_list = []
                self.eval_end_flag = True

        if self.no_swap_tensor(ori_tensor):
            return ori_tensor
        swap_tensor = SwapTensor(ori_tensor, self.layer_name)
        if not self.swap_tensors:
            swap_tensor.first_tensor = True
        # Records the slice tensor status.
        if ori_tensor.storage().size() != ori_tensor.numel():
            swap_tensor.is_slice_tensor = True
            if ori_tensor.storage().data_ptr() not in self.slice_tensor_storage_ptr:
                if self.swap_tensors and self.swap_tensors[0].layer_id != 0:
                    self.slice_tensor_storage_ptr[ori_tensor.storage().data_ptr()] = \
                        [f'{len(self.prefetch_list) - 1}_{len(self.swap_tensors)}']
                else:
                    self.slice_tensor_storage_ptr[ori_tensor.storage().data_ptr()] = \
                        [f'{len(self.prefetch_list)}_{len(self.swap_tensors)}']
            else:
                if self.swap_tensors and self.swap_tensors[0].layer_id != 0:
                    self.slice_tensor_storage_ptr[ori_tensor.storage().data_ptr()].append(
                        f'{len(self.prefetch_list) - 1}_{len(self.swap_tensors)}')
                else:
                    self.slice_tensor_storage_ptr[ori_tensor.storage().data_ptr()].append(
                        f'{len(self.prefetch_list)}_{len(self.swap_tensors)}')

        # Records the same data_ptr tensor status.
        if ori_tensor.storage().data_ptr() in self.data_ptr:
            self.swap_tensors[self.data_ptr[ori_tensor.storage().data_ptr()]].stat = 'h2d'
            swap_tensor.stat = 'd2h'
            swap_tensor.tensor_cpu = self.swap_tensors[self.data_ptr[ori_tensor.storage().data_ptr()]].tensor_cpu
            self.data_ptr[ori_tensor.storage().data_ptr()] = len(self.swap_tensors)
        else:
            self.data_ptr[ori_tensor.storage().data_ptr()] = len(self.swap_tensors)

        swap_tensor.launch_d2h(self.prefetch_stream)
        swap_tensor.stream = self.prefetch_stream
        swap_tensor.layer_id = int(get_layer_id(swap_tensor.layer_name))
        self.swap_tensors.append(swap_tensor)
        self.forward_flag = True
        return swap_tensor

    def unpack_hook(self, swap_tensor):
        if isinstance(swap_tensor, torch.Tensor):
            return swap_tensor
        swap_tensor.wait_h2d_finished(self.prefetch_stream, swap_tensor.last_tensor)
        self.prefetch_list[self.cur_micro_num][swap_tensor.layer_index].remove(swap_tensor)
        # Remove prefetch completed list
        if len(self.prefetch_list[self.cur_micro_num][swap_tensor.layer_index]) == 0:
            self.prefetch_list[self.cur_micro_num].remove(
                self.prefetch_list[self.cur_micro_num][swap_tensor.layer_index])
            self.prefetch_data_ptr_list[self.cur_micro_num].remove(
                self.prefetch_data_ptr_list[self.cur_micro_num][swap_tensor.layer_index])
            self.slice_tensor_storage_ptr_list[self.cur_micro_num].remove(
                self.slice_tensor_storage_ptr_list[self.cur_micro_num][swap_tensor.layer_index])
            if len(self.prefetch_list[self.cur_micro_num]) == 0:
                self.prefetch_list.remove(self.prefetch_list[self.cur_micro_num])
                self.prefetch_data_ptr_list.remove(self.prefetch_data_ptr_list[self.cur_micro_num])
                self.slice_tensor_storage_ptr_list.remove(self.slice_tensor_storage_ptr_list[self.cur_micro_num])
                self.remove_num += 1
                if self.remove_num // self.pp == self.vpp:
                    self.remove_num = 0
        self.forward_flag = False
        return swap_tensor.tensor

    def hook_swap_manager_forward(self, forward_func, layer_name):
        def custom_forward(*args, **kargs):
            self.layer_name = layer_name
            with torch.autograd.graph.saved_tensors_hooks(self.pack_hook, self.unpack_hook):
                return forward_func(*args, **kargs)

        return custom_forward

    def update_slice_tensor_stat(self, swap_tensor):
        if swap_tensor.is_slice_tensor and swap_tensor.storage_data_ptr in self.slice_tensor_storage_ptr:
            _, index = self.slice_tensor_storage_ptr[swap_tensor.storage_data_ptr][0].split('_')
            if swap_tensor != self.swap_tensors[int(index)]:
                swap_tensor.stat = 'host'
                return False
        return True

    def sync_d2h(self, module_name):
        if not self.swap_tensors:
            return
        if self.swap_tensors[0].layer_id <= self.first_layer_id:
            self.first_layer_id = self.swap_tensors[0].layer_id
        elif self.prefetch_list and self.swap_tensors[0].layer_id <= self.prefetch_list[-1][-1][-1].layer_id:
            self.first_layer_id = self.swap_tensors[0].layer_id
        first_resize_tensor = False
        for swap_tensor in self.swap_tensors:
            if self.swap_tensors[0].layer_id > self.first_layer_id and self.prefetch_list:
                swap_tensor.layer_index = len(self.prefetch_list[-1])
            if swap_tensor.layer_id == int(get_layer_id(module_name)) \
                    and swap_tensor.stat == "d2h":
                if not self.update_slice_tensor_stat(swap_tensor):
                    continue
                if not first_resize_tensor:
                    swap_tensor.first_tensor = True
                    first_resize_tensor = True
                # During synchronization, let the first tensor wait for d2h
                swap_tensor.wait_d2h_finished(swap_tensor.stream, swap_tensor.first_tensor)
        self.swap_tensors[-1].last_tensor = True
        if self.swap_tensors[-1].stat == 'host':
            if self.swap_tensors[0].layer_id > self.first_layer_id and self.prefetch_list:
                self.prefetch_list[-1].append(self.swap_tensors)
                self.prefetch_data_ptr_list[-1].append(self.data_ptr)
                self.slice_tensor_storage_ptr_list[-1].append(self.slice_tensor_storage_ptr)
            else:
                self.prefetch_list.append([self.swap_tensors])
                self.prefetch_data_ptr_list.append([self.data_ptr])
                self.slice_tensor_storage_ptr_list.append([self.slice_tensor_storage_ptr])
            self.swap_tensors = []
            self.data_ptr = {}
            self.slice_tensor_storage_ptr = {}
        if self.vpp == 1:
            self.cur_micro_num = 0
        else:
            if not self.remove_num and len(self.prefetch_list) > self.pp:
                self.cur_micro_num = self.pp * (self.vpp - 1)
            elif self.remove_num and self.remove_num % self.pp == 0:
                self.cur_micro_num = self.pp * (self.vpp - 1 - self.remove_num // self.pp)

    def h2d_special_tensor(self, swap_tensor):
        if swap_tensor.is_slice_tensor:
            if swap_tensor.storage_data_ptr in self.slice_tensor_storage_ptr_list[self.cur_micro_num][swap_tensor.layer_index]:
                _, index = self.slice_tensor_storage_ptr_list[self.cur_micro_num][swap_tensor.layer_index][swap_tensor.storage_data_ptr][
                    0].split('_')
                if swap_tensor == self.prefetch_list[self.cur_micro_num][swap_tensor.layer_index][int(index)]:
                    swap_tensor.launch_h2d(self.prefetch_stream, True)
                    del self.slice_tensor_storage_ptr_list[self.cur_micro_num][swap_tensor.layer_index][swap_tensor.storage_data_ptr]
            else:
                swap_tensor.launch_h2d(self.prefetch_stream, False)
        else:
            swap_tensor.launch_h2d(self.prefetch_stream, True)

    def h2d(self, module_name):
        if not self.prefetch_list:
            return
        if self.vpp != 1 and not self.forward_flag:
            self.cur_micro_num = self.pp * (self.vpp - 1 - self.remove_num // self.pp)
        for swap_tensor_list in self.prefetch_list[self.cur_micro_num]:
            for swap_tensor in reversed(swap_tensor_list):
                if swap_tensor.layer_id + self.interval == int(get_layer_id(module_name)) \
                        and swap_tensor.stat == "host" \
                        and swap_tensor.storage_data_ptr in self.prefetch_data_ptr_list[self.cur_micro_num][swap_tensor.layer_index]:
                    del self.prefetch_data_ptr_list[self.cur_micro_num][swap_tensor.layer_index][swap_tensor.storage_data_ptr]
                    # For slice tensors, only the first tensor is resized. Other h2d the tensor size
                    self.h2d_special_tensor(swap_tensor)


def get_swap_prefetch(prefetch_args):
    if SwapPrefetch.swap_prefetch is None:
        SwapPrefetch.swap_prefetch = SwapPrefetch(prefetch_args)

    return SwapPrefetch.swap_prefetch


def pre_forward_hook_func(module_name, prefetch_args):
    def custom_func(module, *args, **kargs):
        get_swap_prefetch(prefetch_args).sync_d2h(module_name)

    return custom_func


def post_backward_hook_func(module_name, prefetch_args):
    def custom_func(module, *args, **kargs):
        get_swap_prefetch(prefetch_args).h2d(module_name)

    return custom_func


# manage activation tensor
def prefetch_tensor(module, name, prefetch_args):
    get_swap_prefetch(prefetch_args).hook_swap_manager_forward(module.forward, name)


# register prefetch before backward, prefetch h2d
def prefetch_register_post_backward_hook(module, name, prefetch_args):
    module.register_backward_hook(post_backward_hook_func(name, prefetch_args))


# register prefetch after forward, sync d2h
def prefetch_register_pre_forward_hook(module, name, prefetch_args):
    module.register_forward_hook(pre_forward_hook_func(name, prefetch_args))
