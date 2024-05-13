import pytest
import torch
import torch_npu
import collections
from ascendspeed import megatron_adaptor
from ascendspeed.core.memory.memory_fragmentation.pluggable_allocator_adpator import change_allocator


@pytest.fixture(scope="class")
def switch_to_pluggable_allocator():
    change_allocator()


@pytest.mark.usefixtures("switch_to_pluggable_allocator")
class TestPluggableAllocatorApi():
    def test_change_alocator(self):
        assert torch_npu.npu.get_allocator_backend() == "pluggable"

    def test_malloc(self):
        npu_tensor = torch.zeros(10, device='npu')
        cpu_tensor = torch.zeros(10)
        assert npu_tensor.cpu().equal(cpu_tensor)
    
    def test_empty_cache(self):
        t1 = torch.zeros(10, device='npu')
        t2 = torch.zeros(20, device='npu')
        t3 = torch.zeros(30, device='npu')
        t4 = torch.zeros(40, device='npu')
        t5 = torch.zeros(50, device='npu')
        del t1
        del t2
        del t3
        del t4
        del t5
        stat_before_empty_cache = torch_npu.npu.memory_stats()
        assert stat_before_empty_cache['reserved_bytes.all.current'] != 0 and stat_before_empty_cache['segment.all.current'] != 0
        torch_npu.npu.empty_cache()
        stat_after_empty_cache = torch_npu.npu.memory_stats()
        assert stat_after_empty_cache['reserved_bytes.all.current'] == 0 and stat_after_empty_cache['segment.all.current'] == 0

    def test_set_memory_fraction(self):
        torch_npu.npu.set_per_process_memory_fraction(0.02)
        with pytest.raises(RuntimeError) as excinfo:
            torch.zeros([536870912], dtype=torch.int32, device='npu')
        assert "NPU out of memory" in str(excinfo.value)
        assert excinfo.type == RuntimeError
    
    def test_memory_stats(self):
        stats = torch_npu.npu.memory_stats()
        assert isinstance(stats, collections.OrderedDict) and len(stats) > 0