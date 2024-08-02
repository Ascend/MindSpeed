import os
import pytest
import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from unit_tests.common import DistributedTest
from mindspeed.ops import ffn
from mindspeed.op_builder import FFNOpBuilder

os.environ["ENABLE_ACLNN"] = "true"
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]
ACTIVATION_TYPE = 'relu'

EXPERT_TOKENS_MODE_NONE = 0
EXPERT_TOKENS_MODE_TOKENS = 1
EXPERT_TOKENS_MODE_TOKENS_INDEX = 2


class FFNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight1, weight2, activation, expert_tokens=None, expert_tokens_index=None,
                bias1=None, bias2=None, scale=None, offset=None, deq_scale1=None, deq_scale2=None,
                antiquant_scale1=None, antiquant_scale2=None, antiquant_offset1=None, antiquant_offset2=None,
                inner_precise=0):
        return ffn.npu_ffn(x, weight1, weight2, activation,
            expert_tokens=expert_tokens, expert_tokens_index=expert_tokens_index,
            bias1=bias1, bias2=bias2, inner_precise=inner_precise)


class TestNPUFFN(DistributedTest):

    @staticmethod
    def relu(x):
        x0 = torch.zeros_like(x)
        return torch.maximum(x, x0)
    
    @staticmethod
    def activation(x):
        return TestNPUFFN.relu(x)

    def mm1_activation_mm2(self, dtype, x, weight1, weight2, bias1, bias2):
        mm1 = x @ weight1 + bias1
        if dtype == torch.bfloat16:
            activation = TestNPUFFN.activation(mm1).to(dtype)
        else:
            activation = TestNPUFFN.activation(mm1.to(dtype))
        return activation.to(torch.float32) @ weight2 + bias2

    def supported_op_exec(self, x, weight1, weight2, bias1, bias2, expert_tokens=None, expert_tokens_index=None):
        dtype = x.dtype
        x = x.to(torch.float32)
        weight1 = weight1.to(torch.float32)
        weight2 = weight2.to(torch.float32)
        bias1 = bias1.to(torch.float32)
        bias2 = bias2.to(torch.float32)
        if expert_tokens is None and expert_tokens_index is None:
            return self.mm1_activation_mm2(dtype, x, weight1, weight2, bias1, bias2)
        y_list = []
        if expert_tokens is not None:
            offset = 0
            for idx, tokens in enumerate(expert_tokens):
                xe = x[offset: offset + tokens]
                w1 = weight1[idx]
                w2 = weight2[idx]
                b1 = bias1[idx]
                b2 = bias2[idx]
                y = self.mm1_activation_mm2(dtype, xe, w1, w2, b1, b2)
                y_list.append(y)
                offset += tokens
        elif expert_tokens_index is not None:
            prev_offset = 0
            for idx, offset in enumerate(expert_tokens_index):
                xe = x[prev_offset: offset]
                w1 = weight1[idx]
                w2 = weight2[idx]
                b1 = bias1[idx]
                b2 = bias2[idx]
                y = self.mm1_activation_mm2(dtype, xe, w1, w2, b1, b2)
                y_list.append(y)
                prev_offset = offset
        result = torch.cat(y_list)
        return result

    def custom_op_exec(self, x, weight1, weight2, *, expert_tokens=None, expert_tokens_index=None,
        bias1=None, bias2=None):
        return ffn.npu_ffn(x, weight1, weight2, ACTIVATION_TYPE,
            expert_tokens=expert_tokens, expert_tokens_index=expert_tokens_index,
            bias1=bias1, bias2=bias2)

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize('tokens_mode', [EXPERT_TOKENS_MODE_NONE, EXPERT_TOKENS_MODE_TOKENS])
    @pytest.mark.parametrize('dtype', [torch.float16])
    @pytest.mark.parametrize('is_graph_mode', [True, False])
    def test_npu_ffn(self, tokens_mode, dtype, is_graph_mode):
        M = 512
        K1 = 256
        N1 = 1024
        K2 = N1
        N2 = K1

        bias_dtype = torch.float16 if dtype == torch.float16 else torch.float32

        expert_tokens = None
        expert_tokens_index = None

        if tokens_mode == EXPERT_TOKENS_MODE_NONE:
            x = torch.empty(M, K1, dtype=dtype).uniform_(-1.0, 1.0)
            weight1 = torch.empty(K1, N1, dtype=dtype).uniform_(-0.1, 0.1)
            weight2 = torch.empty(K2, N2, dtype=dtype).uniform_(-0.1, 0.1)
            bias1 = torch.empty(N1, dtype=bias_dtype).uniform_(-0.1, 0.1)
            bias2 = torch.empty(N2, dtype=bias_dtype).uniform_(-0.1, 0.1)
        elif tokens_mode == EXPERT_TOKENS_MODE_TOKENS:
            E = 8
            x = torch.empty(M, K1, dtype=dtype).uniform_(-1.0, 1.0)
            weight1 = torch.empty(E, K1, N1, dtype=dtype).uniform_(-0.1, 0.1)
            weight2 = torch.empty(E, K2, N2, dtype=dtype).uniform_(-0.1, 0.1)
            bias1 = torch.empty(E, N1, dtype=bias_dtype).uniform_(-0.1, 0.1)
            bias2 = torch.empty(E, N2, dtype=bias_dtype).uniform_(-0.1, 0.1)
            expert_tokens = [64, 64, 64, 64, 64, 64, 64, 64]
            expert_tokens = torch.tensor(expert_tokens, dtype=torch.int64)
        elif tokens_mode == EXPERT_TOKENS_MODE_TOKENS_INDEX:
            E = 8
            x = torch.empty(M, K1, dtype=dtype).uniform_(-1.0, 1.0)
            weight1 = torch.empty(E, K1, N1, dtype=dtype).uniform_(-0.1, 0.1)
            weight2 = torch.empty(E, K2, N2, dtype=dtype).uniform_(-0.1, 0.1)
            bias1 = torch.empty(E, N1, dtype=bias_dtype).uniform_(-0.1, 0.1)
            bias2 = torch.empty(E, N2, dtype=bias_dtype).uniform_(-0.1, 0.1)
            expert_tokens_index = [64, 128, 192, 256, 320, 384, 448, 512]
            expert_tokens_index = torch.tensor(expert_tokens_index, dtype=torch.int64)

        y_golden = self.supported_op_exec(x, weight1, weight2, bias1=bias1, bias2=bias2,
            expert_tokens=expert_tokens, expert_tokens_index=expert_tokens_index)

        x = x.npu()
        weight1 = weight1.npu()
        weight2 = weight2.npu()
        bias1 = bias1.npu()
        bias2 = bias2.npu()

        if expert_tokens is not None:
            expert_tokens = expert_tokens.npu()
        if expert_tokens_index is not None:
            expert_tokens_index = expert_tokens_index.npu()

        if is_graph_mode:
            model = FFNModel().npu()
            model = torch.compile(model, backend=npu_backend, dynamic=True)
            y = model(x, weight1, weight2, ACTIVATION_TYPE, expert_tokens=expert_tokens, 
                expert_tokens_index=expert_tokens_index, bias1=bias1, bias2=bias2)
        else:
            y = self.custom_op_exec(x, weight1, weight2, expert_tokens=expert_tokens,
                expert_tokens_index=expert_tokens_index, bias1=bias1, bias2=bias2)

        assert torch.allclose(y.cpu(), y_golden.to(dtype), rtol=0.005, atol=0.005)
