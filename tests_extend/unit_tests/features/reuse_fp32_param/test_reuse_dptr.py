from unittest import mock

import pytest
import torch
import torch_npu

from mindspeed import megatron_adaptor
from mindspeed.op_builder import AlgorithmOpBuilder
from mindspeed.core.memory.reuse_param.reuse_optimizer import ConvertFp32BF16


class TestReuseDataPtr:
    @staticmethod
    def test_reuse_dptr():
        mindspeed_ops = AlgorithmOpBuilder().load()
        fp32_tensor = torch.randn(10, dtype=torch.float).npu()
        int8_tensor = torch.empty(fp32_tensor.numel() * 4, dtype=torch.int8).npu()
        mindspeed_ops.reuse_data_ptr(int8_tensor, fp32_tensor, 0)
        assert fp32_tensor.data_ptr() == int8_tensor.data_ptr()


class TestReuseFp32Param:
    fp32_tensors = [torch.randn(512, 512, dtype=torch.float32).npu()]
    fp32_truth_tensors = [tensor.clone() for tensor in fp32_tensors]
    bf16_truth_tensors = [tensor.bfloat16().reshape(-1) for tensor in fp32_tensors]

    int32_tensors = [tensor.view(torch.int32).reshape(-1) for tensor in fp32_tensors]
    bf16_fp32_tensors = [tensor.view(torch.bfloat16).reshape(-1) for tensor in fp32_tensors]
    bf16_tensors = [tensor.view(torch.bfloat16).reshape(-1)[tensor.numel():] for tensor in fp32_tensors]
    optimizer = mock.MagicMock()
    optimizer.state = {tensor: {"exp_avg_sq": torch.ones_like(tensor)} for tensor in fp32_tensors}
    reuse_fp32_func = ConvertFp32BF16()

    def test_reuse_fp32_param_deterministic(self):
        self.reuse_fp32_func.fp32_tensors_to_bf16_tensors_deterministic(self.int32_tensors, self.bf16_fp32_tensors, self.fp32_tensors, self.optimizer)
        for reuse, truth in zip(self.bf16_tensors, self.bf16_truth_tensors):
            assert torch.allclose(reuse, truth, rtol=0, atol=0)
        self.reuse_fp32_func.bf16_tensors_to_fp32_tensors_deterministic(self.int32_tensors, self.bf16_fp32_tensors, self.fp32_tensors, self.optimizer)
        for reuse, truth in zip(self.fp32_tensors, self.fp32_truth_tensors):
            assert torch.allclose(reuse, truth, rtol=0, atol=0)
   

    def test_reuse_fp32_param(self):
        # fp32 -> bf16
        self.reuse_fp32_func.fp32_tensors_to_bf16_tensors(int32_tensors=self.int32_tensors, bf16_fp32_tensors=self.bf16_fp32_tensors)
        for reuse, truth in zip(self.bf16_tensors, self.bf16_truth_tensors):
            assert torch.allclose(reuse, truth, rtol=0.005, atol=0.005)
       
        # bf16 -> fp32
        self.reuse_fp32_func.bf16_tensors_to_fp32_tensors(int32_tensors=self.int32_tensors, bf16_fp32_tensors=self.bf16_fp32_tensors)
        for reuse, truth in zip(self.fp32_tensors, self.fp32_truth_tensors):
            assert torch.allclose(reuse, truth, rtol=0, atol=0)
