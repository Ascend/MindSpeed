import os

import pytest
import torch
import torch_npu
# noinspection PyUnusedImports
import mindspeed.megatron_adaptor
from megatron.core.enums import Fp8Recipe

from megatron.training.arguments import parse_args, core_transformer_config_from_args
from megatron.training.global_vars import set_args
from megatron.training.initialize import _set_random_seed
from mindspeed.core.transformer.moe import grouped_gemm_util as gg
from mindspeed.core.transformer.moe.grouped_matmul_util import TensorwiseGMMFunction
from mindspeed.te.pytorch.fp8 import cast_to_fp8, cast_to_fp8_cpu
from mindspeed.te.pytorch.fp8.constants import Format, FormatEnum
from mindspeed.te.pytorch.fp8.fp8 import fp8_autocast
from mindspeed.te.pytorch.fp8.recipes import TensorwiseMatMul, MXFP8MatMul, Float8BlockMatMul
from mindspeed.te.pytorch.module.linear import TEColumnParallelLinear, TERowParallelLinear
from tests_extend.commons import initialize_model_parallel
from tests_extend.unit_tests.common import DistributedTest

WORLD_SIZE = 8


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
    'e4m3': FormatEnum.E4M3,
    'e5m2': FormatEnum.E5M2,
    'hif8': FormatEnum.HIF8
}


def get_config_by_recipe(fp8_recipe, fp8_format, config):
    from mindspeed.te.pytorch.fp8.recipes import Float8CurrentScaling, TEDelayedScaling, MXFP8BlockScaling, \
        Float8BlockScaling
    if fp8_format == "e4m3":
        fp8_format = Format.E4M3
    elif fp8_format == "hybrid":
        fp8_format = Format.HYBRID
    elif fp8_format == 'hif8':
        fp8_format = Format.HIF8
    else:
        raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")
    if fp8_recipe == Fp8Recipe.delayed:
        fp8_recipe = TEDelayedScaling(
            config=config,
            fp8_format=fp8_format,
        )
    elif fp8_recipe == Fp8Recipe.tensorwise:
        fp8_recipe = Float8CurrentScaling(
            config=config,
            fp8_format=fp8_format
        )
    elif fp8_recipe == Fp8Recipe.mxfp8:
        fp8_recipe = MXFP8BlockScaling(
            fp8_format=fp8_format
        )
    elif fp8_recipe == Fp8Recipe.blockwise:
        fp8_recipe = Float8BlockScaling(
            fp8_format=fp8_format
        )
    return fp8_recipe


@pytest.mark.skip(reason='not support for current version')
@pytest.mark.parametrize("fp8_args", [
    ("tensorwise", 'e4m3'),
    ("delayed", 'e4m3'),
    ("blockwise", 'e4m3'),
    ('tensorwise', 'hif8'),
    ('mxfp8', 'e4m3')
])
class TestFP8Model(DistributedTest):

    def test_fp8_column_model(self, fp8_args):
        iteration_num = 10

        os.environ['HCCL_DETERMINISTIC'] = 'True'

        input_size = 8192
        output_size = 8192

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
        recipe_config = get_config_by_recipe(*fp8_args, config)
        initialize_model_parallel(WORLD_SIZE, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        model = ColumnModel(config, input_size, output_size).npu()

        for _ in range(iteration_num):
            inputs = torch.randn([2048, input_size], requires_grad=True, dtype=torch.bfloat16,
                                 device='npu:{}'.format(torch.npu.current_device()))
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
            fp8_context = fp8_autocast(enabled=True, fp8_recipe=recipe_config)
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
        os.environ['HCCL_DETERMINISTIC'] = 'True'

        input_size = 8192
        output_size = 8192

        args = parse_args(None, True)
        args.params_dtype = torch.bfloat16
        args.num_attention_heads = 16
        args.hidden_size = 2048
        args.num_layers = 2
        args.tensor_model_parallel_size = 2
        args.sequence_parallel = True
        args.gradient_accumulation_fusion = False
        set_args(args)
        config = core_transformer_config_from_args(args)
        recipe_config = get_config_by_recipe(*fp8_args, config)
        initialize_model_parallel(WORLD_SIZE, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        model = RowModel(config, input_size, output_size).npu()

        for _ in range(iteration_num):
            inputs = torch.randn([input_size // args.tensor_model_parallel_size, 1024, ], requires_grad=True,
                                 device='npu:{}'.format(torch.npu.current_device()), dtype=torch.bfloat16)
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
            fp8_context = fp8_autocast(enabled=True, fp8_recipe=recipe_config)
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


@pytest.mark.skip(reason='not support for current version')
@pytest.mark.parametrize("fp8_args", [
    ('tensorwise', 'e4m3'),
    ('delayed', 'e4m3'),
    ('blockwise', 'e4m3')
])
class TestFP8Cast(DistributedTest):

    # comparison of cast on NPU and CPU
    def test_fp8_cast(self, fp8_args):
        initialize_model_parallel(WORLD_SIZE, 1)
        recipe, format_str = fp8_args
        fp8_format = FORMAT_MAP[format_str].value

        if fp8_format == FormatEnum.E4M3.value:
            max_val = 448.0
        elif fp8_format == FormatEnum.E5M2.value:
            max_val = 57344.0

        input_size = (128, 256)
        rand_tensor = torch.rand(*input_size, dtype=torch.bfloat16)
        scaled_tensor = rand_tensor * 2 * max_val - max_val

        inputs_npu = scaled_tensor.clone().npu()
        inputs_cpu = scaled_tensor.clone().cpu()

        # NPU FP8 cast
        fp8_npu = cast_to_fp8(inputs_npu, fp8_format)

        # CPU FP8 cast
        fp8_cpu = cast_to_fp8_cpu(inputs_cpu, fp8_format)

        fp8_cpu = fp8_cpu.npu()
        abs_error = torch.abs(fp8_cpu.to(torch.float32) - fp8_npu.to(torch.float32))
        rel_error = abs_error / torch.abs(fp8_cpu.to(torch.float32))
        max_abs_error = torch.max(abs_error)
        max_rel_error = torch.max(rel_error)

        if max_rel_error > 0.0:
            raise ValueError(f"The error of cast exceeds tolerance: {max_rel_error.item()}")


@pytest.mark.skip(reason='not support for current version')
@pytest.mark.parametrize("fp8_args", [
    ('tensorwise', 'e4m3'),
    ('delayed', 'e4m3'),
    ('blockwise', 'e4m3')
])
class TestFP8WithCpu(DistributedTest):

    # comparison of quant_matmul on NPU and CPU
    def test_fp8_quantatmul(self, fp8_args):
        args = parse_args(None, True)
        args.params_dtype = torch.bfloat16
        args.num_attention_heads = 16
        args.hidden_size = 1024
        args.num_layers = 2
        args.tensor_model_parallel_size = 2
        args.sequence_parallel = True
        args.gradient_accumulation_fusion = False
        set_args(args)

        initialize_model_parallel(WORLD_SIZE, 1)
        config = core_transformer_config_from_args(args)
        recipe_config = get_config_by_recipe(*fp8_args, config)

        input_size1 = (256, 128)
        input_size2 = (128, 256)

        bf16_tensor_x1 = torch.randn(*input_size1, dtype=torch.bfloat16)
        bf16_tensor_x2 = torch.randn(*input_size2, dtype=torch.bfloat16)

        # npu
        inputs_npu_x1 = bf16_tensor_x1.clone().npu()
        inputs_npu_x2 = bf16_tensor_x2.clone().npu()
        recipe_npu_x1 = recipe_config.recipe('inputs', recipe_config, input_size1)
        recipe_npu_x2 = recipe_config.recipe('weight', recipe_config, input_size2)

        fp8_npu_x1 = recipe_npu_x1.quantization(inputs_npu_x1)
        fp8_npu_x2 = recipe_npu_x2.quantization(inputs_npu_x2)

        output_npu = fp8_npu_x1.quant_matmul(fp8_npu_x2)

        # cpu
        from mindspeed.te.pytorch.fp8.tensor import Float8TensorCpu
        fp8_cpu_x1 = Float8TensorCpu(
            torch.tensor([]),
            torch.float8_e4m3fn,
            None,
            torch.float32
        )
        fp8_cpu_x2 = Float8TensorCpu(
            torch.tensor([]),
            torch.float8_e4m3fn,
            None,
            torch.float32
        )
        fp8_cpu_x1.from_float8tensor(fp8_npu_x1)
        fp8_cpu_x2.from_float8tensor(fp8_npu_x2)

        output_cpu = fp8_cpu_x1.quant_matmul(fp8_cpu_x2)

        # Compare the error after cast
        fp8_cpu_data_x1 = fp8_cpu_x1.data.npu().to(torch.float32)
        fp8_cpu_data_x2 = fp8_cpu_x2.data.npu().to(torch.float32)
        assert torch.allclose(fp8_npu_x1.data.to(torch.float32), fp8_cpu_data_x1, atol=0.0, rtol=0.0)
        assert torch.allclose(fp8_npu_x2.data.to(torch.float32), fp8_cpu_data_x2, atol=0.0, rtol=0.0)

        # Compare the error after quantmatmul
        assert torch.allclose(output_npu, output_cpu.npu(), atol=0.001, rtol=0.001)


@pytest.mark.skip(reason='not support for current version')
@pytest.mark.parametrize("gmm_quant_func", [
    # MXFP8 暂未支持
    TensorwiseGMMFunction
])
class TestFP8GMM(DistributedTest):
    iter_times = 10
    num_local_experts = 16
    world_size = 8
    hidden_size = 56
    gemm_fusion = False

    def test_gmm_forward(self, gmm_quant_func):
        initialize_model_parallel(self.world_size, 1)
        for _ in range(self.iter_times):
            weight = torch.rand((56, 256), dtype=torch.bfloat16, requires_grad=True).npu()
            x = torch.rand((134, 56), dtype=torch.bfloat16, requires_grad=True).npu()
            tokens_per_expert = torch.tensor([4288, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
                                             dtype=torch.int64).npu()

            weight = weight.view(self.num_local_experts, self.hidden_size, -1)
            normal_x, normal_weight = x.clone(), weight.clone()
            quant_output = gmm_quant_func.apply(x, weight, None, tokens_per_expert)
            quant_bk_dx, quant_bk_dw = torch.autograd.grad(outputs=quant_output.sum(), inputs=(x, weight))
            cp_output, cp_bk_dx, cp_bk_dw = self.compare(normal_weight, normal_x, tokens_per_expert)

            assert torch.allclose(quant_output, cp_output, atol=0.005, rtol=0.005)
            assert torch.allclose(quant_bk_dx, cp_bk_dx, atol=0.005, rtol=0.005)
            assert torch.allclose(quant_bk_dw, cp_bk_dw, atol=0.005, rtol=0.005)

    def compare(self, weight, x, tokens_per_expert):
        # 高精度比对结果
        output = gg.ops.gmm(x, weight, tokens_per_expert, trans_b=False,
                            gemm_fusion=self.gemm_fusion, original_weight=weight)
        dx, dw = torch.autograd.grad(outputs=output.sum(), inputs=(x, weight))
        return output, dx, dw


@pytest.mark.skip(reason='not support for current version')
@pytest.mark.parametrize("fp8_args", [
    (TensorwiseMatMul, 'e4m3'),
    (TensorwiseMatMul, 'hybrid'),
    (TensorwiseMatMul, 'hif8'),
    (MXFP8MatMul, 'e4m3'),
    (MXFP8MatMul, 'hybrid'),
    (Float8BlockMatMul, 'e4m3'),
    (Float8BlockMatMul, 'hybrid'),
])
class TestFP8MM(DistributedTest):
    iter_times = 10
    world_size = 8

    def test_mm(self, fp8_args):
        mm_func, _format = fp8_args
        initialize_model_parallel(self.world_size, 1)
        _args = parse_args(None, True)
        _args.fp8 = _format
        set_args(_args)

        for _ in range(self.iter_times):
            weight = torch.rand((256, 56), dtype=torch.bfloat16, requires_grad=True).npu()
            x = torch.rand((134, 56), dtype=torch.bfloat16, requires_grad=True).npu()

            normal_x, normal_weight = x.clone(), weight.clone()
            quant_output = mm_func.apply(x, weight)
            quant_bk_dx, quant_bk_dw = torch.autograd.grad(outputs=quant_output.sum(), inputs=(x, weight))
            cp_output, cp_bk_dx, cp_bk_dw = self.compare(normal_weight, normal_x)

            assert torch.allclose(quant_output, cp_output, atol=0.005, rtol=0.005)
            assert torch.allclose(quant_bk_dx, cp_bk_dx, atol=0.005, rtol=0.005)
            assert torch.allclose(quant_bk_dw, cp_bk_dw, atol=0.005, rtol=0.005)

    def compare(self, weight, x):
        # 高精度比对结果
        output = torch.matmul(x, weight.t())
        dx, dw = torch.autograd.grad(outputs=output.sum(), inputs=(x, weight))
        return output, dx, dw
