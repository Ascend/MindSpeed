# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import pytest
import torch
import torch_npu
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32

class TestMatmulAdd():
    @pytest.mark.skip(reason='not support for current version')
    def test_matmul_add(self):
        x = torch.rand((1024, 4096), dtype=torch.float16).npu()
        weight = torch.rand((1024, 2048), dtype=torch.float16).npu()
        group_list = torch.tensor([500, 1024], dtype=torch.int64).npu()
        out = torch.rand((8192, 2048), dtype=torch.float32).npu()
        # 分开算子计算结果
        dy = gmm.npu_gmm(x, weight, bias=None, group_list=group_list,group_type=0)
        merged_result =out + dy
        #融合算子计算结果
        x = x.clone().detach()
        weight = weight.clone().detach()
        group_list = group_list.clone().detach()
        out = out.clone().detach()
        npu_groupmatmul_add_fp32(x, weight, group_list, out)
        # 对比
        assert torch.allclose(merged_result, out, rtol=0.005, atol=0.005)
