from mindspeed.core.memory.memory_fragmentation.pluggable_allocator_adpator import load_memory_fragmentation_module


class TestMallocRecorder():
    def test_step_tensor_long_lc(self):
        assert load_memory_fragmentation_module().test_step_tensor_long_lc()

    def test_step_tensor_short_lc(self):
        assert load_memory_fragmentation_module().test_step_tensor_short_lc()

    def test_step_other_branch_long_lc(self):
        assert load_memory_fragmentation_module().test_step_other_branch_long_lc()

    def test_step_other_branch_short_lc(self):
        assert load_memory_fragmentation_module().test_step_other_branch_short_lc()

    def test_change_end_tik(self):
        assert load_memory_fragmentation_module().test_change_end_tik()
