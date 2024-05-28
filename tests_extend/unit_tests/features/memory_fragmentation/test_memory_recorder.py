from mindspeed.core.memory.memory_fragmentation.pluggable_allocator_adpator import load_memory_fragmentation_module


class TestMemoryRecorder():
    def test_setup_tensor_lc(self):
        assert load_memory_fragmentation_module().test_setup_tensor_lc()
    
    def test_forward_tensor_long_lc(self):
        assert load_memory_fragmentation_module().test_forward_tensor_long_lc()
    
    def test_forward_tensor_short_lc(self):
        assert load_memory_fragmentation_module().test_forward_tensor_short_lc()
    
    def test_forward_other_branch_long_lc(self):
        assert load_memory_fragmentation_module().test_forward_other_branch_long_lc()
    
    def test_forward_other_branch_short_lc(self):
        assert load_memory_fragmentation_module().test_forward_other_branch_short_lc()

    def test_change_forward_end_tik(self):
        assert load_memory_fragmentation_module().test_change_forward_end_tik()
