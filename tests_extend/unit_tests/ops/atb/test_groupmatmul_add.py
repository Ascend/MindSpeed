# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import pytest
import torch
import torch_npu
import mindspeed.megatron_adaptor
from mindspeed.ops.gmm import GMMFunction
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestGroupMatmulAdd_float16():
    @pytest.mark.skip(reason='not support for current version')
    def test_matmul_add(self):
        #******************fp16 cases*************************#
        x = torch.rand((1024, 4096), dtype=torch.float16).npu()
        grads_out = torch.rand((1024, 2048), dtype=torch.float16).npu()
        group_list = torch.tensor([500, 1024], dtype=torch.int64).npu()
        out = torch.rand((8192, 2048), dtype=torch.float32).npu()
        weight = torch.rand((2, 4096, 2048), dtype=torch.float16).npu()

        _, dy, _ = GMMFunction.builder.load().npu_gmm_backward([grads_out], [x], [weight], group_list, 0)
        out_single = out + dy[0].view(*out.shape)

        x = x.clone().detach()
        grads_out = grads_out.clone().detach()
        group_list = group_list.clone().detach()
        out = out.clone().detach()
        npu_groupmatmul_add_fp32(x, grads_out, group_list, out)

        assert torch.allclose(out_single, out, rtol=0.005, atol=0.005)


class TestGroupMatmulAdd_bfloat16():
    @pytest.mark.skip(reason='not support for current version')
    def test_matmul_add(self):
        #******************bf16 cases*************************#
        x = torch.rand((8192, 8192), dtype=torch.bfloat16).npu()
        grads_out = torch.rand((8192, 2048), dtype=torch.bfloat16).npu()
        group_list = torch.tensor([1024, 2048, 4096, 8192], dtype=torch.int64).npu()
        out = torch.rand((8192 * 4, 2048), dtype=torch.float32).npu()
        weight = torch.rand((4, 8192, 2048), dtype=torch.bfloat16).npu()

        _, dy, _ = GMMFunction.builder.load().npu_gmm_backward([grads_out], [x], [weight], group_list, 0)
        out_single = out + dy[0].view(*out.shape)

        x = x.clone().detach()
        grads_out = grads_out.clone().detach()
        group_list = group_list.clone().detach()
        out = out.clone().detach()
        npu_groupmatmul_add_fp32(x, grads_out, group_list, out)

        assert torch.allclose(out_single, out, rtol=0.005, atol=0.005)