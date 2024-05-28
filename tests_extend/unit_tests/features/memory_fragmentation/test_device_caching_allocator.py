from mindspeed.core.memory.memory_fragmentation.pluggable_allocator_adpator import load_memory_fragmentation_module



class TestDeviceCachingAllocator():
    def test_round_size(self):
        assert load_memory_fragmentation_module().test_round_size();
