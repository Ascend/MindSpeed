import ctypes

import torch_npu
from mindspeed.op_builder import MemoryFragmentationBuilder


class PluggableAllocatorAdaptor(object):
    MEMORY_FRAGMENTATION_MODULE = None
    def __init__(self):
        pass

def load_memory_fragmentation_module():
    if PluggableAllocatorAdaptor.MEMORY_FRAGMENTATION_MODULE is None:
        PluggableAllocatorAdaptor.MEMORY_FRAGMENTATION_MODULE = MemoryFragmentationBuilder().load()
    return PluggableAllocatorAdaptor.MEMORY_FRAGMENTATION_MODULE

def change_allocator():
    memory_fragmentation_module_path = load_memory_fragmentation_module().__file__

    new_alloc = torch_npu.npu.memory.NPUPluggableAllocator(memory_fragmentation_module_path, 'memory_fragmentation_malloc', 'memory_fragmentation_free')
    torch_npu.npu.memory.change_current_allocator(new_alloc)

    myallocator = ctypes.CDLL(memory_fragmentation_module_path)
    init_fn = ctypes.cast(getattr(myallocator, "memory_fragmentation_init"), ctypes.c_void_p).value
    empty_fn = ctypes.cast(getattr(myallocator, "memory_fragmentation_empty_cache"), ctypes.c_void_p).value
    memory_fraction_fn = ctypes.cast(getattr(myallocator, "memory_fragmentation_memory_fraction"), ctypes.c_void_p).value
    get_device_stats_fn = ctypes.cast(getattr(myallocator, "memory_fragmentation_get_device_stats"), ctypes.c_void_p).value
    reset_peak_status_fn = ctypes.cast(getattr(myallocator, "my_reset_peak_stats"), ctypes.c_void_p).value

    new_alloc.allocator().set_init_fn(init_fn)
    new_alloc.allocator().set_reset_fn(empty_fn)
    new_alloc.allocator().set_memory_fraction_fn(memory_fraction_fn)
    new_alloc.allocator().set_get_device_stats_fn(get_device_stats_fn)
    new_alloc.allocator().set_reset_peak_status_fn(reset_peak_status_fn)