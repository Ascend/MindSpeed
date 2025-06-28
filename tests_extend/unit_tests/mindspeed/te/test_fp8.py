import os

import pytest
import torch
import torch_npu
import mindspeed.megatron_adaptor

from commons import initialize_model_parallel
from megatron.training.arguments import parse_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.global_vars import set_args
from megatron.training.initialize import _set_random_seed
from mindspeed.te.fp8 import fp8_matmul
from mindspeed.te.fp8.constants import Format
from mindspeed.te.fp8.fp8 import fp8_autocast
from mindspeed.te.fp8.metadata import FP8Config
from mindspeed.te.fp8.recipes.block_scaling_recipe import BlockScalingRecipe
from mindspeed.te.fp8.recipes.recipe import RecipeConfig
from mindspeed.te.fp8.recipes.current_scaling_recipe import CurrentScalingRecipe
from mindspeed.te.fp8.recipes.delayed_scaling_recipe import DelayedScalingRecipe
from mindspeed.te.module.linear import TEColumnParallelLinear
from mindspeed.te.module.linear import TERowParallelLinear

from tests_extend.unit_tests.common import DistributedTest


class ColumnModel(torch.nn.Module):
    def __init__(self, config, input_size, output_size, sp=False):
        super().__init__()

        word_size = torch.distributed.get_world_size()
        assert output_size % word_size == 0, 'output size can not div with word size.'
        self.linear = TEColumnParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False
        )

    def forward(self, x):
        return self.linear(x)


class RowModel(torch.nn.Module):
    def __init__(self, config, input_size, output_size, sp=False):
        super().__init__()

        word_size = torch.distributed.get_world_size()
        assert output_size % word_size == 0, 'output size can not div with word size.'
        self.linear = TERowParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=config.init_method,
            input_is_parallel=True,
            bias=False,
            skip_bias_add=False,
            is_expert=False
        )

    def forward(self, x):
        return self.linear(x)


FORMAT_MAP = {
    'e4m3': Format.E4M3,
    'e5m2': Format.E5M2,
    'hif8': Format.HiF8
}


@pytest.mark.skip(reason='not support for current version')
@pytest.mark.parametrize("fp8_args", [
    (CurrentScalingRecipe, 'e4m3'),
    (DelayedScalingRecipe, 'e4m3'),
    (BlockScalingRecipe, 'e4m3'),
    (CurrentScalingRecipe, 'hif8')
])
class TestFP8Model(DistributedTest):
    world_size = 2

    def test_fp8_column_model(self, fp8_args):
        iteration_num = 10
        recipe, format_str = fp8_args
        fp8_format = FORMAT_MAP[format_str]

        fp8_config = FP8Config(default=(recipe, RecipeConfig(block_dim=(2, 3), fp8_format=fp8_format)))
        os.environ['HCCL_DETERMINISTIC'] = 'True'

        input_size = 16
        output_size = 16

        args = parse_args(None, True)
        args.params_dtype = torch.bfloat16
        args.num_attention_heads = 16
        args.hidden_size = 1024
        args.num_layers = 2
        args.tensor_model_parallel_size = 2
        args.sequence_parallel = True
        args.gradient_accumulation_fusion = False
        set_args(args)
        config = core_transformer_config_from_args(args)
        initialize_model_parallel(self.world_size, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        model = ColumnModel(config, input_size, output_size).npu()

        for i in range(iteration_num):
            print('start test {}'.format(i))
            inputs = torch.randn([4, input_size], requires_grad=True, device='npu:{}'.format(torch.npu.current_device()))
            # baseline
            output = model(inputs)[0]
            output.sum().backward()
            baseline = output.clone()
            baseline_wgrad = model.linear.weight.grad.clone()
            baseline_dgrad = inputs.grad.clone()

            # clear grad
            model.zero_grad()
            inputs.grad = None
            # fp8
            fp8_context = fp8_autocast(enabled=True, fp8_config=fp8_config)
            with fp8_context:
                output = model(inputs)[0]
                output.sum().backward()
            fp8 = output.clone()
            fp8_wgrad = model.linear.weight.grad.clone()
            fp8_dgrad = inputs.grad.clone()

            # clear grad
            model.zero_grad()

            torch.cuda.synchronize()
            assert torch.allclose(baseline, fp8, atol=0.005, rtol=0.005)
            assert torch.allclose(baseline_wgrad, fp8_wgrad, atol=0.005, rtol=0.005)
            assert torch.allclose(baseline_dgrad, fp8_dgrad, atol=0.005, rtol=0.005)

    def test_fp8_row_model(self, fp8_args):
        iteration_num = 10
        recipe, format_str = fp8_args
        fp8_format = FORMAT_MAP[format_str]

        fp8_config = FP8Config(default=(recipe, RecipeConfig(block_dim=(2, 2), fp8_format=fp8_format)))
        os.environ['HCCL_DETERMINISTIC'] = 'True'

        input_size = 16
        output_size = 16

        args = parse_args(None, True)
        args.params_dtype = torch.bfloat16
        args.num_attention_heads = 16
        args.hidden_size = 2048
        args.num_layers = 2
        args.tensor_model_parallel_size = 2
        args.sequence_parallel = True
        set_args(args)
        config = core_transformer_config_from_args(args)
        initialize_model_parallel(self.world_size, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        model = RowModel(config, input_size, output_size).npu()

        for i in range(iteration_num):
            print('start test {}'.format(i))
            inputs = torch.randn([4, input_size // args.tensor_model_parallel_size], requires_grad=True, device='npu:{}'.format(torch.npu.current_device()))
            # baseline
            output = model(inputs)[0]
            output.sum().backward()
            baseline = output.clone()
            baseline_wgrad = model.linear.weight.grad.clone()
            baseline_dgrad = inputs.grad.clone()

            # clear grad
            model.zero_grad()
            inputs.grad = None
            # fp8
            fp8_context = fp8_autocast(enabled=True, fp8_config=fp8_config)
            with fp8_context:
                output = model(inputs)[0]
                output.sum().backward()
            fp8 = output.clone()
            fp8_wgrad = model.linear.weight.grad.clone()
            fp8_dgrad = inputs.grad.clone()

            # clear grad
            model.zero_grad()

            torch.cuda.synchronize()
            assert torch.allclose(baseline, fp8, atol=0.005, rtol=0.005)
            assert torch.allclose(baseline_wgrad, fp8_wgrad, atol=0.005, rtol=0.005)
            assert torch.allclose(baseline_dgrad, fp8_dgrad, atol=0.005, rtol=0.005)
