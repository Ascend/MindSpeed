# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import ctypes
import torch_npu

from mindspeed.op_builder import SmartSwapBuilder


class SmartSwapAdaptor(object):
    SMART_SWAP_MODULE = None

    def __init__(self):
        pass


def load_smart_swap_module():
    if SmartSwapAdaptor.SMART_SWAP_MODULE is None:
        SmartSwapAdaptor.SMART_SWAP_MODULE = SmartSwapBuilder().load()
    return SmartSwapAdaptor.SMART_SWAP_MODULE


def change_allocator():
    smart_swap_cpp = load_smart_swap_module()
    smart_swap_module_path = smart_swap_cpp.__file__

    new_alloc = torch_npu.npu.memory.NPUPluggableAllocator(smart_swap_module_path, "gmlake_malloc", "gmlake_free")
    torch_npu.npu.memory.change_current_allocator(new_alloc)

    myallocator = ctypes.CDLL(smart_swap_module_path)
    init_fn = ctypes.cast(getattr(myallocator, "gmlake_init"), ctypes.c_void_p).value
    empty_fn = ctypes.cast(getattr(myallocator, "gmlake_empty_cache"), ctypes.c_void_p).value
    memory_fraction_fn = ctypes.cast(getattr(myallocator, "gmlake_memory_fraction"), ctypes.c_void_p).value
    get_device_stats_fn = ctypes.cast(getattr(myallocator, "gmlake_get_device_stats"), ctypes.c_void_p).value
    reset_peak_stats_fn = ctypes.cast(getattr(myallocator, "gmlake_reset_peak_stats"), ctypes.c_void_p).value
    record_stream_fn = ctypes.cast(getattr(myallocator, "gmlake_record_stream"), ctypes.c_void_p).value
    erase_stream_fn = ctypes.cast(getattr(myallocator, "gmlake_erase_stream"), ctypes.c_void_p).value

    new_alloc.allocator().set_init_fn(init_fn)
    new_alloc.allocator().set_reset_fn(empty_fn)
    new_alloc.allocator().set_memory_fraction_fn(memory_fraction_fn)
    new_alloc.allocator().set_get_device_stats_fn(get_device_stats_fn)
    new_alloc.allocator().set_reset_peak_status_fn(reset_peak_stats_fn)
    new_alloc.allocator().set_record_stream_fn(record_stream_fn)
    new_alloc.allocator().set_erase_stream_fn(erase_stream_fn)
