from enum import IntEnum
from typing import List, Tuple

import torch
import torch_npu
from numpy import mean
from torch.cuda import Event
from megatron.training import print_rank_0

from .adaptive_memory_tool import SingletonBase, CpuTensorCache
from .adaptive_memory_tool import FuncLocationMgr, broadcast_obj
from .adaptive_memory_tool import AdaptiveStepMgr


class SwappableTensorStat(IntEnum):
    HOST = 0
    DEVICE = 1
    D2H = 2
    H2D = 3


class SwappableTensor:
    def __init__(self, tensor, stream, is_prefetch=True):
        self.stream = stream
        self.tensor = tensor
        self.h2d_event = None
        self.is_prefetch = is_prefetch
        self.tensor_cpu = None
        self.storage_size = tensor.storage().size()
        self.stat = SwappableTensorStat.DEVICE
        self.data_ptr = tensor.data_ptr()
        self.storage_data_ptr = tensor.storage().data_ptr()
        self.is_slice_tensor = tensor.storage().size() != tensor.numel()
        self.first_tensor = False
        self.is_allowed_oom_rescue_swap = False
        self.bro_tensors = None
        self.cap_tensor = None  # 和此tensor共享底层storage，并占用整个storage的tensor
        # prefetch
        self.start_pack_event = None
        self.end_pack_event = None
        self.layer_name = "" # 记录tensor在那个module被挂的hook
        self.pack_module_name = None # 记录tensor在那个module被pack出去的
        self.is_firt_same_ptr_tensor = True


    def launch_d2h(self):
        if self.stat != SwappableTensorStat.DEVICE:
            return
        forward_event = torch.npu.Event()
        forward_event.record()
        with torch.no_grad():
            with torch_npu.npu.stream(self.stream):
                self.stream.wait_event(forward_event)
                if self.is_slice_tensor:
                    self.tensor_cpu.copy_(self.tensor, non_blocking=self.is_prefetch)
                else:
                    self.tensor_cpu.storage().copy_(self.tensor.storage(), non_blocking=self.is_prefetch)
                self.stat = SwappableTensorStat.D2H

    def change_stat_to_host(self):
        if self.stat != SwappableTensorStat.D2H:
            return
        self.stat = SwappableTensorStat.HOST

    def launch_h2d(self):
        if self.stat != SwappableTensorStat.HOST:
            return
        with torch.no_grad():
            with torch_npu.npu.stream(self.stream):
                if self.is_slice_tensor:
                    self.tensor.copy_(self.tensor_cpu, non_blocking=self.is_prefetch)
                else:
                    self.tensor.storage().copy_(self.tensor_cpu.storage(), non_blocking=self.is_prefetch)
                if self.h2d_event is not None:
                    self.h2d_event.record()
                self.stat = SwappableTensorStat.H2D


    def change_stat_to_device(self):
        if self.stat != SwappableTensorStat.H2D:
            return
        self.stat = SwappableTensorStat.DEVICE



class SwapManager(metaclass=SingletonBase):
    def __init__(self):
        self.swap_tensors = [] # 存储swap出去的tensor
        self.cpu_tensors = {}
        self.cpu_tensors_h2d_events = {}
        self.prefetch_hooked_modules = []

        self.oom_rescue_device_tensors = {}
        self.oom_rescue_host_tensors = {}
        self.oom_rescue_total_swap_out_size = 0
        self.oom_rescue_hooked_modules = []

        # recording
        self.swap_tensor_in_module = []
        self.layer_name = ""
        self.post_layer_forward_and_pre_layer_backward_hooks = []
        self.forward_time = 0

        self.prefetch_stream = torch_npu.npu.Stream(device=torch.npu.current_device())
        self.oom_rescue_stream = torch_npu.npu.current_stream()

    def get_mean_wait_ms(self, event_pairs):
        time_list = []
        for forward_time in event_pairs:
            start, end = forward_time
            cur_time = start.elapsed_time(end)
            time_list.append(cur_time)
        return mean(time_list)

    def is_need_adjust_module(self, max_overhead_percentage=0.05):
        result = (LayerProfilingHook().get_single_layer_time() - self.forward_time) / self.forward_time > max_overhead_percentage
        result = broadcast_obj(result)
        return result

    def no_swap_tensor(self, ori_tensor):
        if ori_tensor.numel() * ori_tensor.element_size() * 2 < 1024 * 1024:
            return True
        if ori_tensor.grad_fn is None:
            return True
        if ori_tensor.storage().size() == 0:
            return True
        ori_tensor_base = ori_tensor._base
        if ori_tensor_base is not None and ori_tensor_base.dim() >= 5:
            return True
        if ori_tensor_base is not None and ori_tensor_base.grad_fn is None and ori_tensor_base.requires_grad:
            return True
        return False

    def prefetch_pack(self, origin_tensor):
        if self.no_swap_tensor(origin_tensor):
            return origin_tensor
        swap_tensor = SwappableTensor(origin_tensor, self.prefetch_stream)
        if swap_tensor.is_slice_tensor:
            swap_tensor.tensor_cpu = CpuTensorCache().get_cpu_tensor(origin_tensor.shape, origin_tensor.dtype)
            swap_tensor.h2d_event = torch.npu.Event()
        else:
            if swap_tensor.storage_data_ptr not in self.cpu_tensors:
                self.cpu_tensors[swap_tensor.storage_data_ptr] = CpuTensorCache().get_cpu_tensor(origin_tensor.shape, origin_tensor.dtype)
                self.cpu_tensors_h2d_events[swap_tensor.storage_data_ptr] = torch.npu.Event()
                swap_tensor.tensor_cpu = self.cpu_tensors[swap_tensor.storage_data_ptr]
                swap_tensor.h2d_event = self.cpu_tensors_h2d_events[swap_tensor.storage_data_ptr]
            else:
                swap_tensor.tensor_cpu = self.cpu_tensors[swap_tensor.storage_data_ptr]
                swap_tensor.h2d_event = self.cpu_tensors_h2d_events[swap_tensor.storage_data_ptr]
                swap_tensor.stat = SwappableTensorStat.HOST
        swap_tensor.layer_name = self.layer_name

        # 在tensor开始pack的时候插入event
        if AdaptiveStepMgr().is_swap_profiling_step():
            start_pack_event = torch.npu.Event(enable_timing=True)
            start_pack_event.record()
            swap_tensor.start_pack_event = start_pack_event  # 记录tensor开始swap的时间

        swap_tensor.launch_d2h()
        self.swap_tensors.append(swap_tensor)
        if swap_tensor.stat == SwappableTensorStat.D2H:
            self.swap_tensor_in_module.append(swap_tensor)
        return swap_tensor

    def prefetch_unpack(self, swap_tensor):
        if isinstance(swap_tensor, torch.Tensor):
            return swap_tensor

        if swap_tensor.h2d_event:
            torch.cuda.current_stream().wait_event(swap_tensor.h2d_event)
        swap_tensor.change_stat_to_device()
        CpuTensorCache().release_cpu_tensor(swap_tensor.tensor_cpu)
        return swap_tensor.tensor

    def _generate_prefetch_forward_hook(self, origin_forward, layer_name):
        def custom_forward(*args, **kwargs):
            self.layer_name = layer_name
            with torch.autograd.graph.saved_tensors_hooks(self.prefetch_pack, self.prefetch_unpack):
                return origin_forward(*args, **kwargs)
        return custom_forward

    def hook_prefetch_forward(self, module, layer_name):
        module.no_prefetch_hook_forward = module.forward
        self.prefetch_hooked_modules.append(module)
        module.forward = self._generate_prefetch_forward_hook(module.forward, layer_name)

    def reset_prefetch_hooked_modules(self):
        for module in self.prefetch_hooked_modules:
            module.forward = module.no_prefetch_hook_forward
        self.prefetch_hooked_modules.clear()

    def sync_d2h(self, layer_module, is_mark_first_layer):
        if not self.swap_tensors:
            return
        # Wait until the prefetch is complete.
        torch.cuda.current_stream().wait_stream(self.prefetch_stream)
        storage_resized = set()
        for swap_tensor in self.swap_tensors:
            if swap_tensor.stat == SwappableTensorStat.D2H:
                if swap_tensor.storage_data_ptr not in storage_resized:
                    swap_tensor.tensor.storage().resize_(0)
                    storage_resized.add(swap_tensor.storage_data_ptr)
                swap_tensor.change_stat_to_host()

        layer_module.microbatch_swap_tensors_queue.append(self.swap_tensors)
        layer_module.microbatch_cpu_tensors_queue.append(self.cpu_tensors)

        self.swap_tensors = []
        self.cpu_tensors = {}
        self.cpu_tensors_h2d_events = {}
        self.swap_tensor_in_module = []
        if is_mark_first_layer:
            FuncLocationMgr().is_first_layer = False


    def h2d(self, layer_module):
        if not hasattr(layer_module, 'microbatch_swap_tensors_queue'):
            return
        if len(layer_module.microbatch_swap_tensors_queue) == 0 or len(layer_module.microbatch_swap_tensors_queue[-1]) == 0:
            return
        swap_tensors = layer_module.microbatch_swap_tensors_queue.pop(0)
        cpu_tensors = layer_module.microbatch_cpu_tensors_queue.pop(0)
        storage_resized = set()
        self.prefetch_stream.wait_stream(torch.cuda.current_stream())
        for swap_tensor in reversed(swap_tensors):
            if swap_tensor.storage_data_ptr not in storage_resized:
                swap_tensor.tensor.storage().resize_(swap_tensor.storage_size)
                storage_resized.add(swap_tensor.storage_data_ptr)
            if swap_tensor.storage_data_ptr in cpu_tensors:
                cpu_tensors.pop(swap_tensor.storage_data_ptr)
            elif not swap_tensor.is_slice_tensor:
                swap_tensor.stat = SwappableTensorStat.DEVICE
            swap_tensor.launch_h2d()

    def change_oom_rescue_tensors_status_to_allowed_swap(self):
        for wrapped_tensor in self.oom_rescue_device_tensors:
            wrapped_tensor.is_allowed_oom_rescue_swap = True

    def oom_rescue_pack(self, origin_tensor):
        if self.no_swap_tensor(origin_tensor):
            return origin_tensor
        if origin_tensor.grad_fn is None:
            return origin_tensor
        wrapped_tensor = SwappableTensor(origin_tensor, self.oom_rescue_stream, is_prefetch=False)
        self.oom_rescue_device_tensors[wrapped_tensor] = None
        return wrapped_tensor

    def oom_rescue_unpack(self, wrapped_tensor: SwappableTensor):
        if isinstance(wrapped_tensor, torch.Tensor):
            return wrapped_tensor
        if wrapped_tensor in self.oom_rescue_host_tensors:
            self.move_storage_in(wrapped_tensor)
        self.oom_rescue_device_tensors.pop(wrapped_tensor)
        wrapped_tensor.cap_tensor = None
        if wrapped_tensor.bro_tensors is not None:
            wrapped_tensor.bro_tensors.remove(wrapped_tensor)
            wrapped_tensor.bro_tensors = None
        return wrapped_tensor.tensor

    def _generate_oom_rescue_forward_hook(self, origin_forward):
        def custom_forward(*args, **kwargs):
            with torch.autograd.graph.saved_tensors_hooks(self.oom_rescue_pack, self.oom_rescue_unpack):
                return origin_forward(*args, **kwargs)
        return custom_forward

    def hook_oom_rescue_forward(self, module):
        module.no_oom_rescue_hook_forward = module.forward
        self.oom_rescue_hooked_modules.append(module)
        module.forward = self._generate_oom_rescue_forward_hook(module.forward)

    def reset_oom_rescue_hooked_modules(self):
        for module in self.oom_rescue_hooked_modules:
            module.forward = module.no_oom_rescue_hook_forward
        self.oom_rescue_hooked_modules.clear()

    def get_storage_cap_tensor(self, wrapped_tensor: SwappableTensor):
        if wrapped_tensor.cap_tensor is not None:
            return wrapped_tensor.cap_tensor
        storage_tensor = torch.tensor([], dtype=wrapped_tensor.tensor.dtype, device=wrapped_tensor.tensor.device).set_(wrapped_tensor.tensor.storage())
        wrapped_storage_tensor = SwappableTensor(storage_tensor, self.oom_rescue_stream, is_prefetch=False)
        wrapped_storage_tensor.tensor_cpu = torch.empty(storage_tensor.shape, dtype=storage_tensor.dtype, pin_memory=True, device='cpu')
        return wrapped_storage_tensor


    def get_share_storage_tensors(self, wrapped_tensor: SwappableTensor):
        result = set()
        storage_data_ptr = wrapped_tensor.tensor.storage().data_ptr()
        for wt in self.oom_rescue_device_tensors:
            if wt.tensor.storage().data_ptr() == storage_data_ptr:
                result.add(wt)
        return result

    def move_storage_out(self, wrapped_tensor: SwappableTensor):
        if wrapped_tensor not in self.oom_rescue_device_tensors:
            return 0, 0
        storage_size = wrapped_tensor.storage_size * wrapped_tensor.tensor.element_size()
        share_storage_tensors = wrapped_tensor.bro_tensors if wrapped_tensor.bro_tensors is not None else self.get_share_storage_tensors(wrapped_tensor)
        cap_tensor = self.get_storage_cap_tensor(wrapped_tensor)
        cap_tensor.launch_d2h()
        cap_tensor.stat = SwappableTensorStat.HOST
        for wt in share_storage_tensors:
            wt.stat = SwappableTensorStat.HOST
            wt.bro_tensors = share_storage_tensors
            wt.cap_tensor = cap_tensor
            self.oom_rescue_device_tensors.pop(wt)
            self.oom_rescue_host_tensors[wt] = None
        wrapped_tensor.tensor.storage().resize_(0)
        return storage_size, len(share_storage_tensors)

    def move_storage_in(self, wrapped_tensor: SwappableTensor):
        wrapped_tensor.tensor.storage().resize_(wrapped_tensor.storage_size)
        share_storage_tensors = wrapped_tensor.bro_tensors
        wrapped_tensor.cap_tensor.launch_h2d()
        wrapped_tensor.cap_tensor.stat = SwappableTensorStat.DEVICE
        for wt in share_storage_tensors:
            wt.stat = SwappableTensorStat.DEVICE
            self.oom_rescue_host_tensors.pop(wt)
            self.oom_rescue_device_tensors[wt] = None


    def is_exist_tensor_allowed_swap(self):
        for wt in self.oom_rescue_device_tensors:
            if wt.is_allowed_oom_rescue_swap:
                return True
        return False

    def is_exist_tensor_contiguous(self):
        for wt in self.oom_rescue_device_tensors:
            if wt.is_allowed_oom_rescue_swap and wt.tensor.is_contiguous():
                return True
        return False

    def swap_out_by_size(self, size):
        print_rank_0("Need size %d (%fMB)" % (size, size / 1024 / 1024))
        if not self.is_exist_tensor_allowed_swap():
            return False
        swap_size = 0
        swap_num = 0
        only_swap_contiguous_tensor = self.is_exist_tensor_contiguous()
        device_tensors = list(self.oom_rescue_device_tensors.keys())
        for wrapped_tensor in device_tensors:
            if swap_size >= size:
                break
            if not wrapped_tensor.is_allowed_oom_rescue_swap:
                continue
            if only_swap_contiguous_tensor and not wrapped_tensor.tensor.is_contiguous():
                continue

            storage_size, moved_tensor_count = self.move_storage_out(wrapped_tensor)
            swap_size += storage_size
            swap_num += moved_tensor_count

        if swap_size != 0:
            print_rank_0("swap tensor to CPU, tensor num: %d, release NPU memory size: %d (%fMB)" % (
            swap_num, swap_size, swap_size / 1024 / 1024))
            print_rank_0("tensor nums wrap manager for [device: %d, CPU: %d]" % (
                len(self.oom_rescue_device_tensors), len(self.oom_rescue_host_tensors)))
            self.oom_rescue_total_swap_out_size += swap_size
        return True

    def reset_oom_rescue_tensors(self):
        self.oom_rescue_device_tensors.clear()
        self.oom_rescue_host_tensors.clear()

    def reset_all_for_oom_rescue(self):
        self.reset_oom_rescue_tensors()
        self.reset_oom_rescue_hooked_modules()

    def reset_post_layer_forward_and_pre_layer_backward_hooks(self):
        for hook_handle in self.post_layer_forward_and_pre_layer_backward_hooks:
            hook_handle.remove()
        self.post_layer_forward_and_pre_layer_backward_hooks.clear()


def transformer_layer_register_post_forward_hook(module, is_mark_first_layer=False):
    def post_forward_hook(module, *args, **kwargs):
        if not torch.is_grad_enabled():
            return
        if not hasattr(module, 'microbatch_swap_tensors_queue'):
            setattr(module, 'microbatch_swap_tensors_queue', [])
            setattr(module, 'microbatch_cpu_tensors_queue', [])
        SwapManager().sync_d2h(module, is_mark_first_layer)
        SwapManager().change_oom_rescue_tensors_status_to_allowed_swap()
        return

    post_hook = module.register_forward_hook(post_forward_hook)
    SwapManager().post_layer_forward_and_pre_layer_backward_hooks.append(post_hook)


def transformer_layer_register_pre_backward_hook(module):
    def post_forward_hook(module, args, output):
        if not torch.is_grad_enabled():
            return

        def tensor_backward_hook(grad_output):
            SwapManager().h2d(module)
        if isinstance(output, tuple):
            output = output[0]
        output.register_hook(tensor_backward_hook)
    pre_back_hook = module.register_forward_hook(post_forward_hook)
    SwapManager().post_layer_forward_and_pre_layer_backward_hooks.append(pre_back_hook)


class LayerProfilingHook(metaclass=SingletonBase):
    def __init__(self):
        self.hook_handles = []
        self.forward_time_list = []
        self.last_compute_forward_time = None

    def _layer_register_pre_forward_hook(self, module):
        def pre_forward_hook(module, args):
            if AdaptiveStepMgr().is_layer_profiling_step() or AdaptiveStepMgr().is_all_profiling_done():
                start_event = torch.npu.Event(enable_timing=True)
                self.forward_time_list.append([start_event])
                start_event.record()
            else:
                return
        hook_handler = module.register_forward_pre_hook(pre_forward_hook)
        self.hook_handles.append(hook_handler)


    def _layer_register_post_forward_hook(self, module):
        def post_forward_hook(module, args, output):
            if AdaptiveStepMgr().is_layer_profiling_step() or AdaptiveStepMgr().is_all_profiling_done():
                end_event = torch.npu.Event(enable_timing=True)
                self.forward_time_list[-1].append(end_event)
                end_event.record()
            else:
                return
        hook_handler = module.register_forward_hook(post_forward_hook)
        self.hook_handles.append(hook_handler)

    def apply_layer_profiling_hook(self, module):
        self._layer_register_pre_forward_hook(module)
        self._layer_register_post_forward_hook(module)

    def reset_layer_profiling_hook(self):
        for hook_handler in self.hook_handles:
            hook_handler.remove()
        self.hook_handles.clear()

    def get_single_layer_time(self):
        total_time = 0
        forward_cnt = len(self.forward_time_list)
        for event_pair in self.forward_time_list:
            start_event, end_event = event_pair
            tmp_time = start_event.elapsed_time(end_event)
            total_time += tmp_time
        self.last_compute_forward_time = total_time / forward_cnt
        return self.last_compute_forward_time