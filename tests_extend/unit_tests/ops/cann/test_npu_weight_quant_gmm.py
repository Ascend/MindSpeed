import os
import pytest
import torch
import torch_npu
import torchair as tng
import numpy as np

from torchair.configs.compiler_config import CompilerConfig
from mindspeed.ops.weight_quant_gmm import npu_weight_quant_gmm
from mindspeed.ops.weight_quant_gmm import npu_weight_quant_gmm_v2

os.environ["ENABLE_ACLNN"] = "true"
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class WeightQuantGMMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        x, weight, antiquant_scale, antiquant_offset, group_list, group_list_type = args
        if group_list_type == 0:
            y = npu_weight_quant_gmm(x, weight, antiquant_scale=antiquant_scale, antiquant_offset=antiquant_offset,
                                     bias=None, group_list=group_list)
        elif group_list_type == 1:
            y = npu_weight_quant_gmm_v2(x, weight, antiquant_scale=antiquant_scale, antiquant_offset=antiquant_offset,
                                     bias=None, group_list=group_list)
        return y


class TestNPUWeightQuantGMM:

    def supported_op_exec(self, *args):
        x, weight, antiquant_scale, antiquant_offset, group_list = args
        output_dtype = x.dtype
        final_out = []
        num_experts = len(group_list)
        x = x.to(torch.float)
        weight = weight.to(torch.float16)
        x = list(x.split(group_list, dim=0))
        for expert_idx in range(num_experts):
            h = x[expert_idx]
            w = ((weight[expert_idx] + antiquant_offset[expert_idx]) * antiquant_scale[expert_idx]).to(torch.float)
            h_out = (h @ w).to(output_dtype)
            final_out.append(h_out)

        return torch.cat([x for x in final_out], dim=0)

    @pytest.mark.skip(reason='Cann package need update')
    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize('group_list_type', [0, 1])
    @pytest.mark.parametrize('is_graph_mode', [True, False])
    def test_npu_weight_quant_gmm(self, group_list_type, is_graph_mode):
        x = torch.rand(42, 64, dtype=torch.float16).npu()
        weight = torch.randint(-128, 128, (8, 64, 32), dtype=torch.int8).npu()
        antiquant_scale = torch.rand(8, 32, dtype=torch.float16).npu()
        antiquant_offset = torch.rand(8, 32, dtype=torch.float16).npu()

        group_list = [1, 3, 5, 7, 9, 6, 7, 4]
        output = self.supported_op_exec(x, weight, antiquant_scale, antiquant_offset, group_list)
        group_list_value = group_list if group_list_type == 1 else np.cumsum(group_list)
        group_list_tensor = torch.tensor(group_list_value).to(torch.int64).npu()
        model = WeightQuantGMMModel().npu()
        if is_graph_mode:
            model = torch.compile(model, backend=npu_backend, dynamic=True)
        y = model(x, weight, antiquant_scale, antiquant_offset, group_list_tensor, group_list_type)

        assert torch.allclose(y, output, rtol=0.005, atol=0.005)
