import pytest
import torch
import torch_npu

from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestNPUKernelRotaryPositionEmbedding:
    def expect_op_exec(self, x, cos, sin, mode=0):
        x_dtype = x.dtype
        if x_dtype != torch.float:
            x = x.float()
            cos = cos.float()
            sin = sin.float()
        
        if mode == 0:
            xl, xr = torch.chunk(x, 2, dim=-1)
            x_new = torch.cat((-xr, xl), dim=-1)
            res = x * cos + x_new * sin
            return res.to(x_dtype)
        else:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            x_new = torch.stack((-x2, x1), dim=-1)
            x_new = x_new.reshape(x.shape)
            res = x * cos + x_new * sin
            return res.to(x_dtype)

    def custom_op_exec(self, x, cos, sin, mode=0):
        return npu_rotary_position_embedding(x, cos, sin, mode)

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize('dtype', [torch.float, torch.half, torch.bfloat16])
    def test_npu_kernel_rotary_position_embedding_mode_0(self, dtype):
        x = torch.randn(2, 4096, 5, 128, dtype=dtype)
        cos = torch.randn(1, 4096, 1, 128, dtype=dtype)
        sin = torch.randn(1, 4096, 1, 128, dtype=dtype)

        x_npu = x.npu()
        cos_npu = cos.npu()
        sin_npu = sin.npu()
        output = self.expect_op_exec(x_npu, cos_npu, sin_npu, mode=0)
        y = self.custom_op_exec(x_npu, cos_npu, sin_npu, mode=0)
        atol = 0.004 if dtype == torch.bfloat16 else 0.001
        assert torch.allclose(y.cpu(), output.cpu(), rtol=atol, atol=atol)

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize('dtype', [torch.float, torch.half, torch.bfloat16])
    def test_npu_kernel_rotary_position_embedding_mode_1(self, dtype):
        x = torch.randn(2, 4096, 5, 128, dtype=dtype)
        cos = torch.randn(1, 4096, 1, 128, dtype=dtype)
        sin = torch.randn(1, 4096, 1, 128, dtype=dtype)

        x_npu = x.npu()
        cos_npu = cos.npu()
        sin_npu = sin.npu()
        output = self.expect_op_exec(x_npu, cos_npu, sin_npu, mode=1)
        y = self.custom_op_exec(x_npu, cos_npu, sin_npu, mode=1)
        atol = 0.004 if dtype == torch.bfloat16 else 0.001
        assert torch.allclose(y.cpu(), output.cpu(), rtol=atol, atol=atol)
