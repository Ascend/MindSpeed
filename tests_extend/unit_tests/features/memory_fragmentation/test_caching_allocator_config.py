from ascendspeed.core.memory.memory_fragmentation.pluggable_allocator_adpator import load_memory_fragmentation_module


class TestCachingAllocatorConfig():
    def test_parse_args(self):
        assert load_memory_fragmentation_module().test_parse_args()
