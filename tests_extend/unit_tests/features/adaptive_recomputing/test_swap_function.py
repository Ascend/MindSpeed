import time
import torch
from ascendspeed import megatron_adaptor
from ascendspeed.core.memory.adaptive_recomputing.swappable_tensor import SwappableTensor
from ascendspeed.core.memory.adaptive_recomputing.swap_manager import SwapManager

from unit_tests.common import DistributedTest


class TestSwapTensor(DistributedTest):
    world_size = 1

    def test_trans_to_cpu(self):
        tensor_npu = torch.rand((2, 1, 5), dtype=torch.float16, requires_grad=True, device="npu")
        wrapped_tensor = SwappableTensor(tensor_npu)
        key = time.time()
        wrapped_tensor.set_tensor(key, tensor_npu)

        first_mem = torch.npu.memory_allocated()

        assert first_mem != 0
        assert wrapped_tensor.get_location() == "device"

        wrapped_tensor.trans_to_cpu()
        second_mem = torch.npu.memory_allocated()

        assert second_mem == 0
        assert wrapped_tensor.get_location() == "cpu"

    def test_trans_back_to_device(self):
        tensor_npu = torch.rand((2, 2, 5), dtype=torch.float16, requires_grad=True, device="npu")
        wrapped_tensor = SwappableTensor(tensor_npu)
        key = time.time()
        wrapped_tensor.set_tensor(key, tensor_npu)
        first_mem = torch.npu.memory_allocated()

        wrapped_tensor.trans_to_cpu()
        second_mem = torch.npu.memory_allocated()

        assert second_mem == 0
        assert wrapped_tensor.get_location() == "cpu"

        wrapped_tensor.trans_to_device()
        third_mem = torch.npu.memory_allocated()

        assert third_mem == first_mem
        assert wrapped_tensor.get_location() == "device"

    def test_swap_out_by_size(self):
        origin_tensor = torch.rand((3, 1, 5), dtype=torch.float16, requires_grad=True, device="npu")
        wrapped_tensor = SwappableTensor(origin_tensor)
        key = time.time()
        wrapped_tensor.set_tensor(key, origin_tensor)
        SwapManager().device_tensors[key] = wrapped_tensor
        SwapManager().change_manager_tensor_status_to_allowed_swap()
        swap_first_mem = torch.npu.memory_allocated()

        result = SwapManager().swap_out_by_size(swap_first_mem)
        swap_second_mem = torch.npu.memory_allocated()

        assert result is True
        assert swap_second_mem == 0
        assert wrapped_tensor.get_location() == "cpu"

    def test_same_tensor_before_and_after_swapping(self):
        origin_tensor = torch.rand((10, 10, 10), dtype=torch.float16, requires_grad=True, device="npu")
        tmp_tensor = origin_tensor.clone()

        wrapped_tensor = SwappableTensor(origin_tensor)
        key = time.time()
        wrapped_tensor.set_tensor(key, origin_tensor)
        SwapManager().device_tensors[key] = wrapped_tensor
        SwapManager().change_manager_tensor_status_to_allowed_swap()
        first_mem = torch.npu.memory_allocated()

        SwapManager().swap_out_by_size(first_mem)
        second_mem = torch.npu.memory_allocated()
        assert second_mem == first_mem / 2
        assert wrapped_tensor.get_location() == "cpu"

        inner_tensor = SwapManager().unwrap_tensor(wrapped_tensor)
        assert torch.equal(tmp_tensor, inner_tensor)

        third_mem = torch.npu.memory_allocated()
        assert third_mem == first_mem
        assert wrapped_tensor.get_location() == "device"

        assert SwapManager().host_tensors == {}
