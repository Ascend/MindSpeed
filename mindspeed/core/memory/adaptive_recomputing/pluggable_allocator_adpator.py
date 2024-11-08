import ctypes

import torch_npu

from mindspeed.op_builder import AdaptiveRecomputingPluggableAllocatorBuilder


def change_allocator():
    adaptive_recompute_module = AdaptiveRecomputingPluggableAllocatorBuilder().load()
    adaptive_recompute_module_path = adaptive_recompute_module.__file__

    new_alloc = torch_npu.npu.memory.NPUPluggableAllocator(adaptive_recompute_module_path, 'my_malloc', 'my_free')
    torch_npu.npu.memory.change_current_allocator(new_alloc)

    myallocator = ctypes.CDLL(adaptive_recompute_module_path)
    init_fn = ctypes.cast(getattr(myallocator, "my_init"), ctypes.c_void_p).value
    empty_fn = ctypes.cast(getattr(myallocator, "my_empty_cache"), ctypes.c_void_p).value
    get_device_stats_fn = ctypes.cast(getattr(myallocator, "my_get_device_stats"), ctypes.c_void_p).value
    reset_peak_stats_fn = ctypes.cast(getattr(myallocator, "my_reset_peak_stats"), ctypes.c_void_p).value

    new_alloc.allocator().set_init_fn(init_fn)
    new_alloc.allocator().set_reset_fn(empty_fn)
    new_alloc.allocator().set_get_device_stats_fn(get_device_stats_fn)
    new_alloc.allocator().set_reset_peak_status_fn(reset_peak_stats_fn)