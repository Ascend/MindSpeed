import ctypes
import torch

import torch_npu
from mindspeed.op_builder import AdaptiveRecomputingPluggableAllocatorBuilder


class TestPluggableAllocator:
    new_alloc = None
    build_directory = ""

    def test_set_change_allocator(self):
        adaptive_recompute_module = AdaptiveRecomputingPluggableAllocatorBuilder().load()
        TestPluggableAllocator.build_directory = adaptive_recompute_module.__file__

        TestPluggableAllocator.new_alloc = torch_npu.npu.memory.NPUPluggableAllocator(TestPluggableAllocator.build_directory, 'my_malloc', 'my_free')
        torch_npu.npu.memory.change_current_allocator(TestPluggableAllocator.new_alloc)

    def test_set_init_fn(self):
        os_path = TestPluggableAllocator.build_directory
        myallocator = ctypes.CDLL(os_path)
        init_fn = ctypes.cast(getattr(myallocator, "my_init"), ctypes.c_void_p).value

        TestPluggableAllocator.new_alloc.allocator().set_init_fn(init_fn)

    def test_get_device_stats_fn(self):
        os_path = TestPluggableAllocator.build_directory
        myallocator = ctypes.CDLL(os_path)
        get_device_stats_fn = ctypes.cast(getattr(myallocator, "my_get_device_stats"), ctypes.c_void_p).value

        TestPluggableAllocator.new_alloc.allocator().set_get_device_stats_fn(get_device_stats_fn)
        torch_npu.npu.init()
        assert torch.npu.memory_stats_as_nested_dict()["num_alloc_retries"] == 0

    def test_set_reset_peak_stats_fn(self):
        os_path = TestPluggableAllocator.build_directory
        myallocator = ctypes.CDLL(os_path)
        reset_peak_stats_fn = ctypes.cast(getattr(myallocator, "my_reset_peak_stats"), ctypes.c_void_p).value

        TestPluggableAllocator.new_alloc.allocator().set_reset_peak_status_fn(reset_peak_stats_fn)
        torch.npu.reset_peak_memory_stats()
        assert torch.npu.max_memory_allocated() == 0

    def test_set_reset_fn(self):
        os_path = TestPluggableAllocator.build_directory
        myallocator = ctypes.CDLL(os_path)
        empty_fn = ctypes.cast(getattr(myallocator, "my_empty_cache"), ctypes.c_void_p).value

        TestPluggableAllocator.new_alloc.allocator().set_reset_fn(empty_fn)
        torch.npu.empty_cache()
        assert torch.npu.memory_allocated() == 0
