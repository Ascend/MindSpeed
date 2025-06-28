import os

import pytest
import torch
import torch_npu
from torch.nn.parameter import Parameter
import mindspeed.megatron_adaptor

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.training.arguments import parse_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.global_vars import set_args
from megatron.training.initialize import _set_random_seed
from commons import initialize_model_parallel

from mindspeed.te.module.linear import TEColumnParallelLinear
from mindspeed.te.module.linear import TERowParallelLinear
from mindspeed.te.module.ops import CocOps
from mindspeed.te.module.ops.default_ops import DefaultOps
from mindspeed.te.module.ops.mc2_ops import Mc2Ops
from mindspeed.te.module.ops.cockernel_ops import CocKernelOps

from tests_extend.unit_tests.common import DistributedTest


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

        x = torch.randn(seq_size, batch_size, input_size, dtype=dtype).npu()
        w = torch.randn(seq_size, input_size, dtype=dtype).npu()

        output_baseline = DefaultOps.allgather_matmul(x, w.t(), None)
        output_mc2 = Mc2Ops.allgather_matmul(x, w.t(), None)
        output_cockernel = CocKernelOps.allgather_matmul(x, w.t(), None)
        output_coc = CocOps.allgather_matmul(x, w.t(), None)
        assert torch.allclose(output_baseline[0], output_mc2[0], rtol=0.005, atol=0.005)
        assert torch.allclose(output_baseline[0], output_cockernel[0], rtol=0.005, atol=0.005)
        assert torch.allclose(output_baseline[0], output_coc[0], rtol=0.005, atol=0.005)


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

        x = torch.randn(seq_size, batch_size, input_size, dtype=dtype).npu()
        w = torch.randn(seq_size, input_size, dtype=dtype).npu()

        output_baseline, _, _ = DefaultOps.matmul_reduce_scatter(x, w, None)
        output_mc2, _, _ = Mc2Ops.matmul_reduce_scatter(x, w, None)
        output_cockernel, _, _ = CocKernelOps.matmul_reduce_scatter(x, w, None)
        output_coc, _, _ = CocOps.matmul_reduce_scatter(x, w, None)
        assert torch.allclose(output_baseline, output_mc2, rtol=0.005, atol=0.005)
        assert torch.allclose(output_baseline, output_cockernel, rtol=0.005, atol=0.005)
        assert torch.allclose(output_baseline, output_coc, rtol=0.005, atol=0.005)


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

        x = torch.randn(seq_size, batch_size, input_size, dtype=dtype).npu()
        w = torch.randn(seq_size, input_size, dtype=dtype).npu()

        output_baseline, _, _ = DefaultOps.matmul_all_reduce(x, w, None)
        output_mc2, _, _ = Mc2Ops.matmul_all_reduce(x, w, None)
        output_cockernel, _, _ = CocKernelOps.matmul_all_reduce(x, w, None)
        output_coc, _, _ = CocOps.matmul_all_reduce(x, w, None)
        assert torch.allclose(output_baseline, output_mc2, rtol=0.005, atol=0.005)
        assert torch.allclose(output_baseline, output_cockernel, rtol=0.005, atol=0.005)
        assert torch.allclose(output_baseline, output_coc, rtol=0.005, atol=0.005)


class TestTEColumnParallel(DistributedTest):
    world_size = 8

    @pytest.mark.parametrize('coc_type', ['mc2', 'default', 'coc_kernel', 'coc'])
    def test_te_column_parallel(self, coc_type):
        batch_size = 1
        seq_size = 4096
        input_size = 1024
        output_size = 1024
        dtype = torch.bfloat16

        args = parse_args(None, True)
        args.params_dtype = dtype
        args.num_attention_heads = 16
        args.hidden_size = 2048
        args.num_layers = 2
        args.gradient_accumulation_fusion = False
        args.tensor_model_parallel_size = self.world_size
        args.sequence_parallel = True
        args.comm_overlap_type = coc_type
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
            is_expert=False)
        telinear = TEColumnParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False)
        telinear.weight = Parameter(linear.weight.clone())

        outputs = linear(inputs)
        teoutputs = telinear(teinputs)

        outputs[0].sum().backward()
        teoutputs[0].sum().backward()

        assert torch.allclose(outputs[0], teoutputs[0], rtol=0.005, atol=0.005)
        assert torch.allclose(linear.weight.grad, telinear.weight.grad, rtol=0.005, atol=0.005)


class TestTEColumnParallelNoSeq(DistributedTest):
    world_size = 8

    @pytest.mark.parametrize('coc_type', ['mc2', 'default', 'coc_kernel', 'coc'])
    def test_te_column_parallel_no_seq(self, coc_type):
        batch_size = 1
        seq_size = 4096
        input_size = 1024
        output_size = 1024
        dtype = torch.bfloat16

        args = parse_args(None, True)
        args.params_dtype = dtype
        args.num_attention_heads = 16
        args.hidden_size = 2048
        args.num_layers = 2
        args.gradient_accumulation_fusion = False
        args.tensor_model_parallel_size = self.world_size
        args.sequence_parallel = False
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
            is_expert=False)
        telinear = TEColumnParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False)
        telinear.weight = Parameter(linear.weight.clone())

        outputs = linear(inputs)
        teoutputs = telinear(teinputs)

        outputs[0].sum().backward()
        teoutputs[0].sum().backward()

        assert torch.allclose(outputs[0], teoutputs[0], rtol=0.005, atol=0.005)
        assert torch.allclose(linear.weight.grad, telinear.weight.grad, rtol=0.005, atol=0.005)


class TestTERowParallel(DistributedTest):
    world_size = 8

    @pytest.mark.parametrize('coc_type', ['mc2', 'default', 'coc_kernel', 'coc'])
    def test_te_row_parallel(self, coc_type):
        batch_size = 1
        seq_size = 4096
        input_size = 2048
        output_size = 4096
        dtype = torch.bfloat16

        args = parse_args(None, True)
        args.params_dtype = dtype
        args.init_method_std = 0.002
        args.num_attention_heads = 16
        args.hidden_size = 2048
        args.num_layers = 2
        args.gradient_accumulation_fusion = False
        args.tensor_model_parallel_size = self.world_size
        args.sequence_parallel = True
        args.comm_overlap_type = coc_type
        set_args(args)

        config = core_transformer_config_from_args(args)
        initialize_model_parallel(self.world_size, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        inputs = torch.rand(seq_size, batch_size, input_size // args.tensor_model_parallel_size, requires_grad=True, dtype=dtype).npu()
        teinputs = inputs.clone()

        linear = RowParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False)
        telinear = TERowParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False)
        telinear.weight = Parameter(linear.weight.clone())

        outputs = linear(inputs)
        teoutputs = telinear(teinputs)

        y_grad = torch.ones(seq_size // args.tensor_model_parallel_size, batch_size, output_size, dtype=dtype).npu()
        outputs[0].backward(y_grad)
        teoutputs[0].backward(y_grad)

        assert torch.allclose(outputs[0], teoutputs[0], rtol=0.005, atol=0.005)
        assert torch.allclose(linear.weight.grad, telinear.weight.grad, rtol=0.005, atol=0.005)


class TestTERowParallelNoSeq(DistributedTest):
    world_size = 8

    @pytest.mark.parametrize('coc_type', ['mc2', 'default', 'coc_kernel', 'coc'])
    def test_te_row_parallel_no_seq(self, coc_type):
        batch_size = 1
        seq_size = 4096
        input_size = 2048
        output_size = 4096
        dtype = torch.bfloat16

        args = parse_args(None, True)
        args.params_dtype = dtype
        args.init_method_std = 0.002
        args.num_attention_heads = 16
        args.hidden_size = 2048
        args.num_layers = 2
        args.gradient_accumulation_fusion = False
        args.tensor_model_parallel_size = self.world_size
        args.sequence_parallel = False
        args.comm_overlap_type = coc_type
        set_args(args)

        config = core_transformer_config_from_args(args)
        initialize_model_parallel(self.world_size, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        inputs = torch.rand(seq_size, batch_size, input_size // args.tensor_model_parallel_size, requires_grad=True, dtype=dtype).npu()
        teinputs = inputs.clone()

        linear = RowParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False)
        telinear = TERowParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False)
        telinear.weight = Parameter(linear.weight.clone())

        outputs = linear(inputs)
        teoutputs = telinear(teinputs)

        y_grad = torch.ones(seq_size, batch_size, output_size, dtype=dtype).npu()
        outputs[0].backward(y_grad)
        teoutputs[0].backward(y_grad)

        assert torch.allclose(outputs[0], teoutputs[0], rtol=0.005, atol=0.005)
        assert torch.allclose(linear.weight.grad, telinear.weight.grad, rtol=0.005, atol=0.005)