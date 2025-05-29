import pytest
import torch
import torch_npu
from mindspeed import megatron_adaptor
from megatron.legacy.model.transformer import ParallelMLP
from tests_extend.unit_tests.common import DistributedTest
from tests_extend.commons import initialize_model_parallel
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def create_test_args(swiglu=True, use_fused_swiglu=False):
    args = parse_args(None, True)
    args.swiglu = swiglu
    args.use_fused_swiglu = use_fused_swiglu
    return args


class TestNpuSwiglu(DistributedTest):
    world_size = 1

    def patch_ori_op_exec(self, hidden_states):
        args = create_test_args(True, False)
        set_args(args)
        model_parallel_cuda_manual_seed(123)
        self.patch_ori_mlp = ParallelMLP(self.transformer_config).npu().to(hidden_states.dtype)
        hidden_states, output_bias = self.patch_ori_mlp(hidden_states)
        return hidden_states

    def patch_fused_op_exec(self, hidden_states):
        args = create_test_args(True, True)
        set_args(args)
        model_parallel_cuda_manual_seed(123)
        self.patch_fused_mlp = ParallelMLP(self.transformer_config).npu().to(hidden_states.dtype)
        hidden_states, output_bias = self.patch_fused_mlp(hidden_states)
        return hidden_states

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
    def test_npu_swiglu(self, dtype):
        args = parse_args(None, True)
        set_args(args)
        sequence_length, batch_size, hidden_size = 32, 2, 12
        initialize_model_parallel(1, 1)
        self.transformer_config = TransformerConfig(num_layers=2, hidden_size=hidden_size, num_attention_heads=4,
                                                    use_cpu_initialization=False, gated_linear_unit=True)
        hidden_states = torch.randn((sequence_length, batch_size, hidden_size)).npu().to(dtype)
        output_patch_ori = self.patch_ori_op_exec(hidden_states)
        output_patch_fused = self.patch_fused_op_exec(hidden_states)
        tol = 0.004 if dtype == torch.bfloat16 else 0.001
        assert torch.allclose(output_patch_ori, output_patch_fused, rtol=tol, atol=tol)
