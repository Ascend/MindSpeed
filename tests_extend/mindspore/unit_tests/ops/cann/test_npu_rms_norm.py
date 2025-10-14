from types import SimpleNamespace
import pytest
import torch
from torch import nn
import torch_npu
from mindspeed import megatron_adaptor
from megatron.legacy.model.rms_norm import RMSNorm
from megatron.training.global_vars import set_args

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def create_test_args(use_fused_rmsnorm=False):
    # Set dummy values for the args
    args = SimpleNamespace()
    args.use_fused_rmsnorm = use_fused_rmsnorm
    args.use_nd_matmul = False
    return args


class TestNpuRmsNorm:

    def supported_op_exec(self, x, dim, eps=1e-6):
        ori_dtype = x.dtype
        for _ in range(30):
            x = x.to(torch.float)
            weight = nn.Parameter(torch.ones(dim)).npu().to(ori_dtype)
            norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            x = norm.to(ori_dtype) * weight
        return x

    def patch_ori_op_exec(self, x, dim, eps=1e-6, sequence_parallel=False):
        args = create_test_args(False)
        set_args(args)
        patch_rms_norm = RMSNorm(dim, eps, sequence_parallel).npu().to(x.dtype)
        for _ in range(30):
            x = patch_rms_norm(x)
        return x

    def patch_fused_op_exec(self, x, dim, eps=1e-6, sequence_parallel=False):
        args = create_test_args(True)
        set_args(args)
        patch_rms_norm = RMSNorm(dim, eps, sequence_parallel).npu().to(x.dtype)
        for _ in range(30):
            x = patch_rms_norm(x)
        return x

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
    def test_npu_rms_norm(self, dtype):
        batch, seq, hidden_size = 6, 60, 1024
        x = torch.randn((batch, seq, hidden_size)).npu().to(dtype)
        output_golden = self.supported_op_exec(x, hidden_size)
        output_patch_ori = self.patch_ori_op_exec(x, hidden_size)
        output_patch_fused = self.patch_fused_op_exec(x, hidden_size)
        tol = 0.004 if dtype == torch.bfloat16 else 0.001
        assert torch.allclose(output_golden, output_patch_ori, rtol=tol, atol=tol)
        assert torch.allclose(output_golden, output_patch_fused, rtol=tol, atol=tol)
