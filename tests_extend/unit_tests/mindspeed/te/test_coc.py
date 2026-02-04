import pytest

import torch
import torch_npu

from torch.nn.parameter import Parameter
import mindspeed.megatron_adaptor
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.training.arguments import parse_args, core_transformer_config_from_args
from megatron.training.global_vars import set_args
from megatron.training.initialize import _set_random_seed
from mindspeed.te.pytorch.fp8.metadata import FP8Metadata
from mindspeed.te.pytorch.module.linear import TEColumnParallelLinear, TERowParallelLinear
from mindspeed.te.pytorch.module.ops.default_ops import DefaultOps
from mindspeed.te.pytorch.module.ops.mc2_ops import Mc2Ops
from tests_extend.commons import initialize_model_parallel
from tests_extend.unit_tests.common import DistributedTest
from tests_extend.unit_tests.utils import multi_compare


class TestAllgatherMatmul(DistributedTest):
    world_size = 8

    def test_allgather_matmul(self):
        batch_size = 1
        seq_size = 4096
        input_size = 1024
        dtype = torch.bfloat16

        args = parse_args(None, True)
        args.params_dtype = dtype
        set_args(args)
        initialize_model_parallel(self.world_size, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        x = torch.randn(seq_size, batch_size, input_size, dtype=dtype) * 5
        w = torch.randn(seq_size, input_size, dtype=dtype) * 5
        fp8_meta = FP8Metadata(["inputs", "weight", "grads"])
        output_baseline = self.allgather_matmul(x, w).view(-1)
        output_default = DefaultOps.allgather_matmul(x.npu(), w.npu(), None, fp8_meta, transpose=(False, True))
        output_mc2 = Mc2Ops.allgather_matmul(x.npu(), w.npu(), None, fp8_meta, transpose=(False, True))
        output_default = output_default[0].view(-1).cpu()
        output_mc2 = output_mc2[0].view(-1).cpu()
        assert multi_compare(output_mc2, output_baseline, output_default, f"{torch.npu.current_device()}") != "FAIL"

    def allgather_matmul(self, input_, weight):
        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * self.world_size
        total_input = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
        group = torch.distributed.new_group(list(range(self.world_size)), backend="gloo")
        torch.distributed._all_gather_base(total_input, input_.contiguous(), group=group, async_op=False)
        return torch.matmul(total_input, weight.t())


class TestMatmulReduceScatter(DistributedTest):
    world_size = 8

    def test_matmul_reduce_sactter(self):
        batch_size = 1
        seq_size = 4096
        input_size = 1024
        dtype = torch.bfloat16

        args = parse_args(None, True)
        args.params_dtype = dtype
        set_args(args)
        initialize_model_parallel(self.world_size, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        x = torch.randn(seq_size, batch_size, input_size, dtype=dtype) * 5
        w = torch.randn(seq_size, input_size, dtype=dtype) * 5
        fp8_meta = FP8Metadata(["inputs", "weight", "grads"])
        output_baseline = self.reduce_scatter(x, w).view(-1)
        output_default, _, _ = DefaultOps.matmul_reduce_scatter(x.npu(), w.npu(), None, fp8_meta)
        output_mc2, _, _ = Mc2Ops.matmul_reduce_scatter(x.npu(), w.npu(), None, fp8_meta)
        output_default = output_default.view(-1).cpu()
        output_mc2 = output_mc2.view(-1).cpu()
        assert multi_compare(
            output_mc2,
            output_baseline,
            output_default,
            f"{torch.npu.current_device()}",
            "l0"
        ) != "FAIL"

    def reduce_scatter(self, x, w):
        output_ = torch.matmul(x, w.t())
        dim_size = list(output_.size())
        dim_size[0] = dim_size[0] // self.world_size
        output = torch.empty(dim_size, dtype=output_.dtype)
        group = torch.distributed.new_group(list(range(self.world_size)), backend="gloo")
        torch.distributed._reduce_scatter_base(output, output_.contiguous(), group=group)
        return output


class TestMatmulAllReduce(DistributedTest):
    world_size = 8

    def test_matmul_all_reduce(self):
        batch_size = 1
        seq_size = 4096
        input_size = 1024
        dtype = torch.bfloat16

        args = parse_args(None, True)
        args.params_dtype = dtype
        set_args(args)
        initialize_model_parallel(self.world_size, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        x = torch.randn(seq_size, batch_size, input_size, dtype=dtype) * 5
        w = torch.randn(seq_size, input_size, dtype=dtype) * 5
        fp8_meta = FP8Metadata(["inputs", "weight", "grads"])
        output_baseline = self.matmul_all_reduce(x, w).view(-1)
        output_default, _, _ = DefaultOps.matmul_all_reduce(x.npu(), w.npu(), None, fp8_meta)
        output_mc2, _, _ = Mc2Ops.matmul_all_reduce(x.npu(), w.npu(), None, fp8_meta)
        output_default = output_default.view(-1).cpu()
        output_mc2 = output_mc2.view(-1).cpu()
        assert multi_compare(
            output_mc2,
            output_baseline,
            output_default,
            f"{torch.npu.current_device()}",
            "l0"
        ) != "FAIL"

    def matmul_all_reduce(self, x, w):
        output = torch.matmul(x, w.t())
        group = torch.distributed.new_group(list(range(self.world_size)), backend="gloo")
        torch.distributed.all_reduce(output, group=group)
        return output


class TestTEColumnParallel(DistributedTest):
    world_size = 8

    @pytest.mark.parametrize("use_ascend_mc2", [True, False])
    @pytest.mark.parametrize("limit_args", [
        (torch.bfloat16, 0.005, 0.005)
    ])
    def test_te_column_parallel(self, use_ascend_mc2, limit_args):
        batch_size = 1
        seq_size = 4096
        input_size = 1024
        output_size = 1024
        dtype, rtol, atol = limit_args

        args = parse_args(None, True)
        args.params_dtype = dtype
        args.num_attention_heads = 16
        args.hidden_size = 2048
        args.num_layers = 2
        args.gradient_accumulation_fusion = False
        args.tensor_model_parallel_size = self.world_size
        args.sequence_parallel = True
        args.use_ascend_mc2 = use_ascend_mc2
        args.transformer_impl = "transformer_engine"
        set_args(args)

        config = core_transformer_config_from_args(args)
        initialize_model_parallel(self.world_size, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        inputs = torch.rand(batch_size, seq_size, input_size, requires_grad=True, dtype=dtype).npu()
        teinputs = inputs.clone()

        linear = ColumnParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )
        telinear = TEColumnParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )
        telinear.weight = Parameter(linear.weight.clone())

        outputs = linear(inputs)
        teoutputs = telinear(teinputs)

        outputs[0].sum().backward()
        teoutputs[0].sum().backward()

        assert torch.allclose(outputs[0], teoutputs[0], rtol=rtol, atol=atol)
        assert torch.allclose(linear.weight.grad, telinear.weight.grad, rtol=rtol, atol=atol)


class TestTEColumnParallelNoSeq(DistributedTest):
    world_size = 8

    @pytest.mark.parametrize("use_ascend_mc2", [True, False])
    @pytest.mark.parametrize("limit_args", [
        (torch.bfloat16, 0.005, 0.005)
    ])
    def test_te_column_parallel_no_seq(self, use_ascend_mc2, limit_args):
        batch_size = 1
        seq_size = 4096
        input_size = 1024
        output_size = 1024
        dtype, rtol, atol = limit_args

        args = parse_args(None, True)
        args.params_dtype = dtype
        args.num_attention_heads = 16
        args.hidden_size = 2048
        args.num_layers = 2
        args.gradient_accumulation_fusion = False
        args.tensor_model_parallel_size = self.world_size
        args.sequence_parallel = False
        args.use_ascend_mc2 = use_ascend_mc2
        args.transformer_impl = "transformer_engine"
        set_args(args)

        config = core_transformer_config_from_args(args)
        initialize_model_parallel(self.world_size, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        inputs = torch.rand(batch_size, seq_size, input_size, requires_grad=True, dtype=dtype).npu()
        teinputs = inputs.clone()

        linear = ColumnParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )
        telinear = TEColumnParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )
        telinear.weight = Parameter(linear.weight.clone())

        outputs = linear(inputs)
        teoutputs = telinear(teinputs)

        outputs[0].sum().backward()
        teoutputs[0].sum().backward()

        assert torch.allclose(outputs[0], teoutputs[0], rtol=rtol, atol=atol)
        assert torch.allclose(linear.weight.grad, telinear.weight.grad, rtol=rtol, atol=atol)


class TestTERowParallel(DistributedTest):
    world_size = 8

    @pytest.mark.parametrize("use_ascend_mc2", [True, False])
    @pytest.mark.parametrize("limit_args", [
        (torch.bfloat16, 0.005, 0.005)
    ])
    def test_te_row_parallel(self, use_ascend_mc2, limit_args):
        batch_size = 1
        seq_size = 4096
        input_size = 2048
        output_size = 4096
        dtype, rtol, atol = limit_args

        args = parse_args(None, True)
        args.params_dtype = dtype
        args.init_method_std = 0.002
        args.num_attention_heads = 16
        args.hidden_size = 2048
        args.num_layers = 2
        args.gradient_accumulation_fusion = False
        args.tensor_model_parallel_size = self.world_size
        args.sequence_parallel = True
        args.use_ascend_mc2 = use_ascend_mc2
        args.transformer_impl = "transformer_engine"
        set_args(args)

        config = core_transformer_config_from_args(args)
        initialize_model_parallel(self.world_size, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        inputs = torch.rand(
            seq_size, batch_size, input_size // args.tensor_model_parallel_size, requires_grad=True, dtype=dtype
        ).npu()
        teinputs = inputs.clone()

        linear = RowParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
        )
        telinear = TERowParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
        )
        telinear.weight = Parameter(linear.weight.clone())

        outputs = linear(inputs)
        teoutputs = telinear(teinputs)

        y_grad = torch.ones(seq_size // args.tensor_model_parallel_size, batch_size, output_size, dtype=dtype).npu()
        outputs[0].backward(y_grad)
        teoutputs[0].backward(y_grad)

        assert torch.allclose(outputs[0], teoutputs[0], rtol=rtol, atol=atol)
        assert torch.allclose(linear.weight.grad, telinear.weight.grad, rtol=rtol, atol=atol)


class TestTERowParallelNoSeq(DistributedTest):
    world_size = 8

    @pytest.mark.parametrize("use_ascend_mc2", [True, False])
    @pytest.mark.parametrize("limit_args", [
        (torch.bfloat16, 0.005, 0.005)
    ])
    def test_te_row_parallel_no_seq(self, use_ascend_mc2, limit_args):
        batch_size = 1
        seq_size = 4096
        input_size = 2048
        output_size = 4096
        dtype, rtol, atol = limit_args

        args = parse_args(None, True)
        args.params_dtype = dtype
        args.init_method_std = 0.002
        args.num_attention_heads = 16
        args.hidden_size = 2048
        args.num_layers = 2
        args.gradient_accumulation_fusion = False
        args.tensor_model_parallel_size = self.world_size
        args.sequence_parallel = False
        args.use_ascend_mc2 = use_ascend_mc2
        args.transformer_impl = "transformer_engine"
        set_args(args)

        config = core_transformer_config_from_args(args)
        initialize_model_parallel(self.world_size, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        inputs = torch.rand(
            seq_size, batch_size, input_size // args.tensor_model_parallel_size, requires_grad=True, dtype=dtype
        ).npu()
        teinputs = inputs.clone()

        linear = RowParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
        )
        telinear = TERowParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
        )
        telinear.weight = Parameter(linear.weight.clone())

        outputs = linear(inputs)
        teoutputs = telinear(teinputs)

        y_grad = torch.ones(seq_size, batch_size, output_size, dtype=dtype).npu()
        outputs[0].backward(y_grad)
        teoutputs[0].backward(y_grad)

        assert torch.allclose(outputs[0], teoutputs[0], rtol=rtol, atol=atol)
        assert torch.allclose(linear.weight.grad, telinear.weight.grad, rtol=rtol, atol=atol)
