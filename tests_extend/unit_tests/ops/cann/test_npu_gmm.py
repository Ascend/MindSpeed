import pytest
import torch
import torch_npu

from mindspeed.ops.gmm import npu_gmm

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestNPUGMM:

    def supported_op_exec(self, x, weight, group_list, group_type):
        final_out = []
        num_experts = weight.shape[0]
        if group_type == 0:
            x = list(x.split(group_list, dim=0))
        elif group_type == 2:
            x = list(x.split(group_list, dim=-1))
            weight = list(weight.split(group_list, dim=0))
        for expert_idx in range(num_experts):
            h = x[expert_idx]
            h_out = h @ weight[expert_idx]
            final_out.append(h_out)
        
        return torch.cat([x for x in final_out], dim=0)

    def custom_op_exec(self, x, weight, bias, group_list, group_type):
        return npu_gmm(x, weight, bias=bias, group_list=group_list, group_type=group_type)

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_npu_gmm_1(self):
        x = torch.randn(32, 64, dtype=torch.float16)
        weight = torch.randn(8, 64, 32, dtype=torch.float16)

        group_list = [1, 2, 3, 4, 5, 6, 7, 4]
        group_list_index = [1, 3, 6, 10, 15, 21, 28, 32]
        group_type = 0

        x_npu = x.npu()
        weight_npu = weight.npu()
        output = self.supported_op_exec(x_npu, weight_npu, group_list, group_type)
        y = self.custom_op_exec(x_npu, weight_npu, None, group_list_index, group_type)
        assert torch.allclose(y, output, rtol=0.005, atol=0.005)

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_npu_gmm_2(self):
        x = torch.randn(32, 64, dtype=torch.float16)
        weight = torch.randn(8, 64, 32, dtype=torch.float16)

        group_list = [1, 2, 3, 4, 5, 6, 7, 4]
        group_list_index = torch.tensor([1, 3, 6, 10, 15, 21, 28, 32]).npu()
        group_type = 0

        x_npu = x.npu()
        weight_npu = weight.npu()
        output = self.supported_op_exec(x_npu, weight_npu, group_list, group_type)
        y = self.custom_op_exec(x_npu, weight_npu, None, group_list_index, group_type)
        assert torch.allclose(y, output, rtol=0.005, atol=0.005)
