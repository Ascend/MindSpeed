import torch
import torch_npu
import ctypes
from mindspeed.core.memory.memory_fragmentation.pluggable_allocator_adpator import load_memory_fragmentation_module

os_path = load_memory_fragmentation_module().__file__
new_alloc = torch_npu.npu.memory.NPUPluggableAllocator(os_path, 'memory_fragmentation_malloc', 'memory_fragmentation_free')
torch_npu.npu.memory.change_current_allocator(new_alloc)

myallocator = ctypes.CDLL(os_path)
init_fn = ctypes.cast(getattr(myallocator, "memory_fragmentation_init"), ctypes.c_void_p).value
empty_fn = ctypes.cast(getattr(myallocator, "memory_fragmentation_empty_cache"), ctypes.c_void_p).value
memory_fraction_fn = ctypes.cast(getattr(myallocator, "memory_fragmentation_memory_fraction"), ctypes.c_void_p).value
get_device_stats_fn = ctypes.cast(getattr(myallocator, "memory_fragmentation_get_device_stats"), ctypes.c_void_p).value

new_alloc.allocator().set_init_fn(init_fn)
new_alloc.allocator().set_reset_fn(empty_fn)
new_alloc.allocator().set_memory_fraction_fn(memory_fraction_fn)
new_alloc.allocator().set_get_device_stats_fn(get_device_stats_fn)

load_memory_fragmentation_module().precise_match_start()
load_memory_fragmentation_module().precise_match_end()


def report_memory(name):
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated:{}'.format(torch.cuda.memory_allocated() / mega_bytes)
    print(string)


new_tensor = torch.zeros(10, device='npu')
report_memory("report_memory")
torch_npu.npu.set_per_process_memory_fraction(0.5)

del new_tensor
