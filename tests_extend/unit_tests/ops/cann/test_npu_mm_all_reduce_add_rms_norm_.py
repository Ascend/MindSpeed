import os
import pytest
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group, ReduceOp
import torch_npu
from unit_tests.common import DistributedTest
from mindspeed.ops.npu_mm_all_reduce_add_rms_norm_ import npu_mm_all_reduce_add_rms_norm_
DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestNPUInplaceMMAllReduceAddRmsNorm(DistributedTest):
    world_size = 2

    def supported_op_exec(self, x1, x2, hcomm_info, residual_shape, residual, gamma, epsilon, dequant_scale):
        output_npu = torch_npu.npu_mm_all_reduce_base(x1=x1,
                                                      x2=x2,
                                                      hcom=hcomm_info,
                                                      reduce_op="sum",
                                                      dequant_scale=dequant_scale)
        output_npu = output_npu.reshape(residual_shape)
        output_golden, rstd_golden, y_golden = torch_npu.npu_add_rms_norm(output_npu,
                                                                          residual,
                                                                          gamma,
                                                                          epsilon)
        return y_golden, output_golden

    def custom_op_exec(self, x1, x2, hcomm_info, residual, gamma, epsilon, dequant_scale):
        y, norm_out = npu_mm_all_reduce_add_rms_norm_(x1=x1,
                                                    x2=x2,
                                                    residual=residual,
                                                    gamma=gamma,
                                                    hcom=hcomm_info,
                                                    epsilon=epsilon,
                                                    dequant_scale=dequant_scale)
        return y, norm_out

    def get_hcomm_info(self, n, i):
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device('npu')).get_hccl_comm_name(i)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(i)
        return hcomm_info

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.int8])
    def test_npu_mm_all_reduce_add_rms_norm_(self, dtype):
        rank = int(os.environ["LOCAL_RANK"])
        hcomm_info = self.get_hcomm_info(self.world_size, rank)
        print("current device: {}, local rank = {}, hcomm_info = {}".format(torch_npu.npu.current_device(), rank, hcomm_info))
        b, s, k, n = 4, 1024, 1024, 8192
        x1_shape = (b, s, k)
        x2_shape = (k, n)
        x1 = torch.rand(x1_shape)
        x2 = torch.rand(x2_shape)
        x1_npu = x1.npu().to(dtype)
        x2_npu = x2.npu().to(dtype)

        if dtype is torch.int8:
            dequant_scale_shape = (n)
            dequant_scale = torch.rand(dequant_scale_shape)
            dequant_scale_npu = dequant_scale.npu().to(torch.bfloat16)
            residual_shape = (b, s, n)
            gamma_shape = (n)
            residual = torch.rand(residual_shape)
            gamma = torch.rand(gamma_shape)
            residual_npu = residual.npu().to(torch.bfloat16)
            gamma_npu = gamma.npu().to(torch.bfloat16)
        else:
            dequant_scale_npu = None
            residual_shape = (b, s, n)
            gamma_shape = (n)
            residual = torch.rand(residual_shape)
            gamma = torch.rand(gamma_shape)
            residual_npu = residual.npu().to(dtype)
            gamma_npu = gamma.npu().to(dtype)

        epsilon = 0.000001
        support_y, support_output = self.supported_op_exec(x1_npu, x2_npu, hcomm_info, residual_shape, residual_npu, gamma_npu, epsilon, dequant_scale_npu)
        y, output = self.custom_op_exec(x1_npu, x2_npu, hcomm_info, residual_npu, gamma_npu, epsilon, dequant_scale_npu)

        for y_i, support_y_i in zip(y, support_y):
            assert torch.allclose(y_i, support_y_i, rtol=0.005, atol=0.005)
        for output_i, support_output_i in zip(output, support_output):
            assert torch.allclose(output_i, support_output_i, rtol=0.005, atol=0.005)