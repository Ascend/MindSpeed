import os
import pytest
import torch
import torch_npu
import torchair as tng

from torchair.configs.compiler_config import CompilerConfig
from mindspeed.ops.gmm import npu_gmm
from mindspeed.ops.gmm import npu_gmm_v2

os.environ["ENABLE_ACLNN"] = "true"
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]

GROUP_LIST_VECTOR_CUMSUM = 0
GROUP_LIST_TENSOR_CUMSUM = 1
GROUP_LIST_TENSOR_COUNT = 2


class GMMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, group_list, group_type, group_list_type):
        if group_list_type == 0:
            y = npu_gmm(x, weight, bias=None, group_list=group_list, group_type=group_type)
        elif group_list_type == 1:
            y = npu_gmm_v2(x, weight, bias=None, group_list=group_list, group_type=group_type)
        return y


class TestNPUGMM:

    def supported_op_exec(self, x, weight, group_list, group_type):
        final_out = []
        num_experts = len(group_list)
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

    def custom_op_exec(self, x, weight, group_list, group_type, group_list_type):
        if group_list_type == 0:
            y = npu_gmm(x, weight, bias=None, group_list=group_list, group_type=group_type)
        elif group_list_type == 1:
            y = npu_gmm_v2(x, weight, bias=None, group_list=group_list, group_type=group_type)
        return y

    @pytest.mark.skip(reason='aclnnGroupedMatmulV4 is not in this cann')
    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize('gropu_list_mode', [GROUP_LIST_VECTOR_CUMSUM, GROUP_LIST_TENSOR_CUMSUM, GROUP_LIST_TENSOR_COUNT])
    @pytest.mark.parametrize('is_graph_mode', [True, False])
    def test_npu_gmm(self, gropu_list_mode, is_graph_mode):
        weight = torch.randn(8, 64, 32, dtype=torch.float16)
        weight_npu = weight.npu()
        group_type = 0

        if gropu_list_mode == GROUP_LIST_TENSOR_COUNT:
            x = torch.randn(42, 64, dtype=torch.float16)
            x_npu = x.npu()
            group_list = [1, 3, 5, 7, 9, 6, 7, 4]
            output = self.supported_op_exec(x_npu, weight_npu, group_list, group_type)
            group_list_index = torch.tensor([1, 3, 5, 7, 9, 6, 7, 4]).to(torch.int64).npu()
            if is_graph_mode:
                model = GMMModel().npu()
                model = torch.compile(model, backend=npu_backend, dynamic=True)
                y = model(x_npu, weight_npu, group_list_index, group_type, 1)
            else:
                y = self.custom_op_exec(x_npu, weight_npu, group_list_index, group_type, 1)
        else:
            x = torch.randn(32, 64, dtype=torch.float16)
            x_npu = x.npu()
            group_list = [1, 2, 3, 4, 5, 6, 7, 4]
            output = self.supported_op_exec(x_npu, weight_npu, group_list, group_type)
            if gropu_list_mode == GROUP_LIST_VECTOR_CUMSUM:
                group_list_index = [1, 3, 6, 10, 15, 21, 28, 32]
                y = self.custom_op_exec(x_npu, weight_npu, group_list_index, group_type, 0)
            elif gropu_list_mode == GROUP_LIST_TENSOR_CUMSUM:
                group_list_index = torch.tensor([1, 3, 6, 10, 15, 21, 28, 32]).to(torch.int64).npu()
                if is_graph_mode:
                    model = GMMModel().npu()
                    model = torch.compile(model, backend=npu_backend, dynamic=True)
                    y = model(x_npu, weight_npu, group_list_index, group_type, 0)
                else:
                    y = self.custom_op_exec(x_npu, weight_npu, group_list_index, group_type, 0)

        assert torch.allclose(y, output, rtol=0.005, atol=0.005)


    @pytest.mark.skip(reason='aclnnGroupedMatmulV4 is not in this cann')
    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize('gropu_list_mode', [GROUP_LIST_VECTOR_CUMSUM, GROUP_LIST_TENSOR_CUMSUM, GROUP_LIST_TENSOR_COUNT])
    def test_npu_gmm_group_type_2(self, gropu_list_mode):
        group_type = 2

        if gropu_list_mode == GROUP_LIST_TENSOR_COUNT:
            weight = torch.randn(42, 64, dtype=torch.float16)
            weight_npu = weight.npu()
            x = torch.randn(42, 48, dtype=torch.float16)
            x_npu = x.npu().transpose(-1, -2)
            group_list = [9, 0, 0, 7, 9, 6, 7, 4]
            output = self.supported_op_exec(x_npu, weight_npu, group_list, group_type)
            group_list_index = torch.tensor([9, 0, 0, 7, 9, 6, 7, 4]).to(torch.int64).npu()
            y = self.custom_op_exec(x_npu, weight_npu, group_list_index, group_type, 1)
        else:
            weight = torch.randn(32, 64, dtype=torch.float16)
            weight_npu = weight.npu()
            x = torch.randn(32, 48, dtype=torch.float16)
            x_npu = x.npu().transpose(-1, -2)
            group_list = [1, 2, 3, 4, 5, 6, 7, 4]
            output = self.supported_op_exec(x_npu, weight_npu, group_list, group_type)
            if gropu_list_mode == GROUP_LIST_VECTOR_CUMSUM:
                group_list_index = [1, 3, 6, 10, 15, 21, 28, 32]
                y = self.custom_op_exec(x_npu, weight_npu, group_list_index, group_type, 0)
            elif gropu_list_mode == GROUP_LIST_TENSOR_CUMSUM:
                group_list_index = torch.tensor([1, 3, 6, 10, 15, 21, 28, 32]).to(torch.int64).npu()
                y = self.custom_op_exec(x_npu, weight_npu, group_list_index, group_type, 0)

        assert torch.allclose(y.reshape(-1, 64), output, rtol=0.005, atol=0.005)