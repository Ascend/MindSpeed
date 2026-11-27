# Copyright (c) 2026, MindSpeed contributors. All rights reserved.
import pytest
import torch
import torch_npu
import mindspeed.megatron_adaptor
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args

from mindspeed.te.pytorch.fp8.constants import Format
from mindspeed.te.pytorch.fp8.fp8 import fp8_autocast
from mindspeed.te.pytorch.fp8.recipes import TEDelayedScaling
from mindspeed.te.pytorch.module.linear import TEColumnParallelLinear
from tests_extend.commons import initialize_model_parallel
from tests_extend.unit_tests.common import DistributedTest


class TestColumnParallelSeqBias(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize('fp8_enabled', [False, True])
    @pytest.mark.parametrize('bias_kind', ['none', 'zero', 'nonzero'])
    @pytest.mark.parametrize('skip_bias_add', [False, True])
    def test_forward(self, fp8_enabled, bias_kind, skip_bias_add):
        args = parse_args(ignore_unknown_args=True)
        args.gradient_accumulation_fusion = False
        args.use_ascend_mc2 = False
        set_args(args)
        initialize_model_parallel(self.world_size, 1)
        try:
            config = TransformerConfig(
                num_layers=1,
                hidden_size=64,
                num_attention_heads=2,
                num_query_groups=2,
                tensor_model_parallel_size=self.world_size,
                sequence_parallel=True,
                params_dtype=torch.bfloat16,
                use_cpu_initialization=True,
                perform_initialization=False,
            )
            model = TEColumnParallelLinear(
                64, 64, config=config, init_method=torch.nn.init.normal_,
                gather_output=False, bias=bias_kind != 'none',
                skip_bias_add=skip_bias_add, is_expert=False,
            ).npu()
            rank = parallel_state.get_tensor_model_parallel_rank()
            with torch.no_grad():
                model.weight.zero_()
                if model.bias is not None:
                    model.bias.zero_()
                    if bias_kind == 'nonzero':
                        model.bias[::2] = 8
                        model.bias[1::2] = -4

            # Zero weights isolate bias addition from FP8 scaling/rounding error.
            full_input = ((torch.arange(32 * 64, device='npu') % 4 + 1) / 4).to(torch.bfloat16)
            full_input = full_input.reshape(32, 64)
            local_input = full_input.chunk(self.world_size, dim=0)[rank].contiguous()
            reference = torch.nn.functional.linear(full_input, model.weight, model.bias)
            recipe = TEDelayedScaling(fp8_format=Format.HYBRID, config=config)
            with fp8_autocast(enabled=fp8_enabled, fp8_recipe=recipe):
                output, output_bias = model(local_input)

            if skip_bias_add and model.bias is not None:
                torch.testing.assert_close(output, reference - model.bias, rtol=0, atol=0)
                torch.testing.assert_close(output_bias, model.bias, rtol=0, atol=0)
                output = output + output_bias
            else:
                assert output_bias is None
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
        finally:
            parallel_state.destroy_model_parallel()

    @pytest.mark.parametrize('fp8_enabled', [False, True])
    @pytest.mark.parametrize('bias_kind', ['none', 'zero', 'nonzero'])
    @pytest.mark.parametrize('skip_bias_add', [False, True])
    def test_nonzero_weight_forward_backward(self, fp8_enabled, bias_kind, skip_bias_add):
        # Removing the forward bias addition must fail the nonzero/non-skip
        # cases; changing any backward result must fail a full-tensor assertion.
        args = parse_args(ignore_unknown_args=True)
        args.gradient_accumulation_fusion = False
        args.use_ascend_mc2 = False
        set_args(args)
        initialize_model_parallel(self.world_size, 1)
        try:
            config = TransformerConfig(
                num_layers=1, hidden_size=64, num_attention_heads=2,
                num_query_groups=2, tensor_model_parallel_size=self.world_size,
                sequence_parallel=True, params_dtype=torch.bfloat16,
                use_cpu_initialization=True, perform_initialization=False,
            )
            model = TEColumnParallelLinear(
                64, 64, config=config, init_method=torch.nn.init.normal_,
                gather_output=False, bias=bias_kind != 'none',
                skip_bias_add=skip_bias_add, is_expert=False,
            ).npu()
            with torch.no_grad():
                model.weight.fill_(0.125)
                if model.bias is not None:
                    model.bias.zero_()
                    if bias_kind == 'nonzero':
                        model.bias[::2] = 8
                        model.bias[1::2] = -4

            # For the first delayed-scale amax=1, E4M3 quantization maps
            # [.125, .25, .5, 1] to [56, 112, 224, 448], all exactly
            # representable. Merely choosing BF16-exact inputs is not enough.
            rank = parallel_state.get_tensor_model_parallel_rank()
            full_input = torch.tensor(
                [0.125, 0.25, 0.5, 1.0], device='npu', dtype=torch.bfloat16,
            ).repeat(32, 16)
            local_input = full_input.chunk(self.world_size, dim=0)[rank]
            local_input = local_input.contiguous().detach().requires_grad_(True)
            recipe = TEDelayedScaling(fp8_format=Format.HYBRID, config=config)
            with fp8_autocast(enabled=fp8_enabled, fp8_recipe=recipe):
                output, output_bias = model(local_input)
            if skip_bias_add and model.bias is not None:
                torch.testing.assert_close(output, torch.full_like(output, 3.75), rtol=0, atol=0)
                torch.testing.assert_close(output_bias, model.bias, rtol=0, atol=0)
                output = output + output_bias
            else:
                assert output_bias is None
            expected_output = torch.full_like(output, 3.75)
            if bias_kind == 'nonzero':
                expected_output[:, ::2] = 11.75
                expected_output[:, 1::2] = -0.25
            torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
            output.backward(torch.full_like(output, 0.25))

            # Hand-derived global TP2 derivatives, not a second invocation of
            # the patched implementation: dX=64*.125*.25=2 after TP reduction;
            # dW=32*.25*x=[1,2,4,8] repeated; db=32*.25=8.
            assert local_input.grad is not None
            assert model.weight.grad is not None
            torch.testing.assert_close(local_input.grad, torch.full_like(local_input, 2), rtol=0, atol=0)
            expected_dw = torch.tensor([1, 2, 4, 8], device='npu', dtype=model.weight.dtype)
            expected_dw = expected_dw.repeat(16).expand_as(model.weight)
            torch.testing.assert_close(model.weight.grad, expected_dw, rtol=0, atol=0)
            if model.bias is not None:
                assert model.bias.grad is not None
                torch.testing.assert_close(model.bias.grad, torch.full_like(model.bias, 8), rtol=0, atol=0)
        finally:
            parallel_state.destroy_model_parallel()
