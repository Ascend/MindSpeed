import pytest
import torch
import torch_npu
from mindspeed import megatron_adaptor

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from mindspeed.core.tensor_parallel.unaligned_layers.adaptor import UnalignedColumnParallelLinearAdaptor, \
    UnalignedRowParallelLinearAdaptor
from tests_extend.unit_tests.common import DistributedTest
from tests_extend.commons import initialize_model_parallel
from tests_extend.commons import set_random_seed


def set_unaligned_linear_args(args):
    args.tensor_model_parallel_size = 8
    args.seed = 2025
    args.seq_len = 180
    args.batch_size = 8
    args.sequence_parallel = True
    args.num_query_groups = None


class TestUnalignedLinear(DistributedTest):
    world_size = 8

    @pytest.mark.skip(reason="The local verification is successful. The CI environment problem is to be located")
    def test_UnalignedColumnParallelLinear(self):
        args = parse_args(None, True)
        set_unaligned_linear_args(args)
        set_args(args)
        transformer_config = TransformerConfig(num_layers=1,
                                               tensor_model_parallel_size=1,
                                               sequence_parallel=False,
                                               hidden_size=160,
                                               num_attention_heads=18,
                                               use_cpu_initialization=True)
        transformer_config.sequence_parallel = args.sequence_parallel
        transformer_config.tensor_model_parallel_size = args.tensor_model_parallel_size
        transformer_config.gradient_accumulation_fusion = False
        initialize_model_parallel(args.tensor_model_parallel_size, 1)
        set_random_seed(args.seed)
        input_size = transformer_config.hidden_size
        output_size = transformer_config.hidden_size
        linear_layer = UnalignedColumnParallelLinearAdaptor(input_size,
                                                            output_size,
                                                            keep_master_weight_for_test=True,
                                                            init_method=transformer_config.init_method,
                                                            config=transformer_config).half().npu()
        setattr(linear_layer.weight, 'main_grad', linear_layer.weight.clone())
        loss_weight = torch.rand([args.seq_len,
                                  transformer_config.hidden_size // args.tensor_model_parallel_size]
                                 ).half().npu()
        input_ = torch.rand(args.batch_size, args.seq_len, input_size).half().npu()
        output = linear_layer(input_)
        gather_list = [torch.zeros(input_.shape).half().npu() for _ in range(self.world_size)]
        torch.distributed.all_gather(gather_list, input_, group=parallel_state.get_tensor_model_parallel_group())
        gather_res = torch.concat(gather_list, dim=0)
        output_naive = torch.matmul(gather_res, linear_layer.weight.t())
        assert torch.allclose(output_naive, output[0], rtol=0.005, atol=0.005)
        loss = torch.mul(output[0], loss_weight).sum()
        # Backward
        loss.backward()
        dLdY = loss_weight
        dLdA = torch.matmul(dLdY.t(), gather_res).sum(dim=0)
        ones = torch.ones(args.seq_len, args.batch_size * self.world_size).half().npu()
        dLdb = torch.matmul(ones.t(), dLdY).sum(dim=0).view(-1)
        assert torch.allclose(dLdA, linear_layer.weight.grad, rtol=0.005, atol=0.005)
        assert torch.allclose(dLdb, linear_layer.bias.grad, rtol=0.005, atol=0.005)

    def test_UnalignedRowParallelLinear(self):
        args = parse_args(None, True)
        set_unaligned_linear_args(args)
        set_args(args)
        transformer_config = TransformerConfig(num_layers=1,
                                               tensor_model_parallel_size=1,
                                               sequence_parallel=False,
                                               hidden_size=160,
                                               num_attention_heads=18,
                                               use_cpu_initialization=True)
        transformer_config.sequence_parallel = args.sequence_parallel
        transformer_config.tensor_model_parallel_size = args.tensor_model_parallel_size
        transformer_config.gradient_accumulation_fusion = False
        initialize_model_parallel(args.tensor_model_parallel_size, 1)
        set_random_seed(args.seed)
        input_size = transformer_config.hidden_size
        output_size = transformer_config.hidden_size
        linear_layer = UnalignedRowParallelLinearAdaptor(input_size,
                                                         output_size,
                                                         keep_master_weight_for_test=True,
                                                         bias=True, input_is_parallel=True,
                                                         skip_bias_add=False,
                                                         init_method=transformer_config.init_method,
                                                         config=transformer_config).half().npu()
        setattr(linear_layer.weight, 'main_grad', linear_layer.weight.clone())
        loss_weight = torch.rand([args.seq_len, output_size]).half().npu()
        input_ = torch.rand(args.batch_size, args.seq_len,
                            transformer_config.hidden_size // args.tensor_model_parallel_size)
        input_ = input_.half().npu()
        output = linear_layer(input_)
        loss = torch.mul(output[0], loss_weight).sum()
        # Backward
        loss.backward()
        dLdY = loss_weight
        rank = torch.distributed.get_rank()
        res = torch.matmul(input_, linear_layer.weight.npu().T)
        tensor = torch.empty(args.batch_size // self.world_size, args.seq_len, output_size)
        tensor = tensor.half().npu()
        scatter_list = list(torch.chunk(res, chunks=self.world_size, dim=0))
        torch.distributed.reduce_scatter(tensor, scatter_list, group=parallel_state.get_tensor_model_parallel_group())
        assert torch.allclose(tensor, output[0], rtol=0.005, atol=0.005)
        x = torch.chunk(input_, chunks=self.world_size, dim=0)[rank]
        torch.distributed.all_reduce(x)
        dLdA = torch.matmul(dLdY.t(), x).sum(dim=0)
        ones = torch.ones(args.seq_len, args.batch_size // self.world_size).half().npu()
        dLdb = torch.matmul(ones.t(), dLdY).sum(dim=0).view(-1)
        assert torch.allclose(dLdA, linear_layer.weight.grad, rtol=0.005, atol=0.005)
        assert torch.allclose(dLdb, linear_layer.bias.grad, rtol=0.005, atol=0.005)
