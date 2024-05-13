import pytest
import torch
import torch_npu

from ascendspeed.op_builder import AlgorithmOpBuilder
from unit_tests.common import DistributedTest


class TestReuseDataPtr(DistributedTest):
    world_size = 1

    def test_reuse_dptr(self):
        ascendspeed_ops = AlgorithmOpBuilder().load()
        fp32_tensor = torch.randn(10, dtype=torch.float).npu()
        int8_tensor = torch.empty(fp32_tensor.numel() * 4, dtype=torch.int8).npu()
        ascendspeed_ops.reuse_data_ptr(int8_tensor, fp32_tensor, 0)
        assert fp32_tensor.data_ptr() == int8_tensor.data_ptr()
