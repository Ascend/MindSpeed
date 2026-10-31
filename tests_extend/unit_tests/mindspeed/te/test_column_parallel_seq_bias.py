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
