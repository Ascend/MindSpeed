# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import pytest
import torch

from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_groupmatmul_add_without_atb(dtype):
    torch.manual_seed(295)
    x = torch.randn(1024, 128, dtype=dtype).npu()
    dy = torch.randn(1024, 64, dtype=dtype).npu()
    groups = torch.tensor([500, 1024], dtype=torch.int64).npu()
    grad = torch.randn(256, 64, dtype=torch.float32).npu()
    expected = grad.cpu()
    for index, (start, end) in enumerate([(0, 500), (500, 1024)]):
        expected[index * 128 : (index + 1) * 128] += x[start:end].cpu().float().t() @ dy[start:end].cpu().float()
    assert npu_groupmatmul_add_fp32(x, dy, groups, grad) is None
    torch.testing.assert_close(grad.cpu(), expected, rtol=0.0003, atol=0.0003)
