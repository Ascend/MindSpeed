# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import pytest
import torch
import torch_npu
from mindspeed.ops.npu_matmul_add import npu_matmul_add


class TestMatmulAdd():
    @pytest.mark.skip(reason='not support for current version')
    def test_matmul_add(self):
        for i in range(1000):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            x = torch.rand((4096, 8192), dtype=torch.float16).to(device)
            weight = torch.rand((4096, 8192), dtype=torch.float16).to(device)
            c = torch.rand((8192, 8192), dtype=torch.float).to(device)
            # 分开算子计算结果
            product = torch.mm(x.T, weight)
            result = product + c
            # 融合算子计算结果
            x = x.clone().detach()
            weight = weight.clone().detach()
            c = c.clone().detach()
            npu_matmul_add(weight, x, c)

            # 对比
            assert torch.allclose(result, c, rtol=0.005, atol=0.005)
