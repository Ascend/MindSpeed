import os
import pytest
import torch
import torch_npu
import torchair as tng
import numpy as np

from torchair.configs.compiler_config import CompilerConfig
from mindspeed.ops.quant_gmm import npu_quant_gmm
from mindspeed.ops.quant_gmm import npu_quant_gmm_v2

os.environ["ENABLE_ACLNN"] = "true"
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class QuantGMMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        x, weight, scale, per_token_scale, group_list, group_list_type, output_dtype = args
        if group_list_type == 0:
            y = npu_quant_gmm(x, weight, scale=scale, per_token_scale=per_token_scale, bias=None,
                              group_list=group_list, output_dtype=output_dtype)
        elif group_list_type == 1:
            y = npu_quant_gmm_v2(x, weight, scale=scale, per_token_scale=per_token_scale, bias=None,
                                 group_list=group_list, output_dtype=output_dtype)
        return y


class TestNPUQuantGMM:

    def supported_op_exec(self, *args):
        x, weight, scale, per_token_scale, group_list, output_dtype = args
        final_out = []
        num_experts = len(group_list)
        x = x.to(torch.float)
        weight = weight.to(torch.float)
        x = list(x.split(group_list, dim=0))
        per_token_scale = list(per_token_scale.split(group_list, dim=0))
        for expert_idx in range(num_experts):
            h = x[expert_idx]
            h_out = h @ weight[expert_idx]
            h_out = (h_out * scale[expert_idx].reshape(1, -1) * per_token_scale[expert_idx].reshape(-1, 1)).to(output_dtype)
            final_out.append(h_out)

        return torch.cat([x for x in final_out], dim=0)

    @pytest.mark.skip(reason='Cann package need update')
    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize('group_list_type', [0, 1])
    @pytest.mark.parametrize('is_graph_mode', [True, False])
    def test_npu_quant_gmm(self, group_list_type, is_graph_mode):
        x = torch.randint(-128, 128, (42, 64), dtype=torch.int8).npu()
        weight = torch.randint(-128, 128, (8, 64, 32), dtype=torch.int8).npu()
        scale = torch.rand(8, 32, dtype=torch.float32).npu()
        per_token_scale = torch.rand(42, dtype=torch.float32).npu()

        group_list = [1, 3, 5, 7, 9, 6, 7, 4]
        output = self.supported_op_exec(x, weight, scale, per_token_scale, group_list, torch.float16)
        group_list_value = group_list if group_list_type == 1 else np.cumsum(group_list)
        group_list_tensor = torch.tensor(group_list_value).to(torch.int64).npu()
        model = QuantGMMModel().npu()
        if is_graph_mode:
            model = torch.compile(model, backend=npu_backend, dynamic=True)
        y = model(x, weight, scale, per_token_scale, group_list_tensor, group_list_type, torch.float16)

        assert torch.allclose(y, output, rtol=0.005, atol=0.005)
