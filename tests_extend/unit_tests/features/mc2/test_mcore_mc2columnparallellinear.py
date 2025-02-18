import pytest
import torch
import torch_npu
from mindspeed import megatron_adaptor
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig
from mindspeed.core.tensor_parallel.ascend_turbo.initialize import initialize_cfg_from_args
from unit_tests.common import DistributedTest
from commons import initialize_model_parallel
from commons import set_random_seed


def set_mc2_args(args):
    args.tensor_model_parallel_size = 8
    args.use_unpad = False
    args.seed = 2024
    args.seq_len = 128
    args.input_size_coeff = 128
    args.batch_size = 8
    args.optimize_recomp_communication_level = True
    args.sequence_parallel = True
    args.use_cp_send_recv_overlap = False
    args.vocab_size = 512
    return args


class TestMC2(DistributedTest):
    world_size = 8

    def test_mcore_mc2_column_parallel_linear(self):
        args = parse_args(None, True)
        args = set_mc2_args(args)
        set_args(args)

        transformer_config = TransformerConfig(num_layers=1,
                                               hidden_size=12,
                                               num_attention_heads=4,
                                               use_cpu_initialization=True)
        transformer_config.sequence_parallel = args.sequence_parallel
        initialize_model_parallel(args.tensor_model_parallel_size, 1)
        input_size = args.input_size_coeff * args.tensor_model_parallel_size
        output_size = args.vocab_size
        input_ = torch.rand(args.seq_len, args.batch_size, input_size).half().npu()
        loss_weight = torch.rand([8, 64]).half().npu()

        # get output_weight
        linear_layer = ColumnParallelLinear(input_size,
                                            output_size,
                                            keep_master_weight_for_test=True,
                                            init_method=transformer_config.init_method,
                                            config=transformer_config).half().npu()
        setattr(linear_layer.weight, 'main_grad', linear_layer.weight.clone())
        output_weight_mc2_close = linear_layer.weight
        output_weight_mc2_open = linear_layer.weight

        # close mc2 forward and backward
        set_random_seed(args.seed)
        linear_layer_mc2_close = ColumnParallelLinear(input_size,
                                                      output_size,
                                                      keep_master_weight_for_test=True,
                                                      init_method=transformer_config.init_method,
                                                      config=transformer_config).half().npu()
        output_mc2_close, _ = linear_layer_mc2_close(input_, output_weight_mc2_close)
        loss_mc2_close = torch.mul(output_mc2_close, loss_weight).sum()
        loss_mc2_close.backward()
        output_weight_grad_mc2_close = output_weight_mc2_close.grad

        # open mc2 forward and backward
        initialize_cfg_from_args(args)
        set_random_seed(args.seed)
        linear_layer_mc2_open = ColumnParallelLinear(input_size,
                                                     output_size,
                                                     keep_master_weight_for_test=True,
                                                     init_method=transformer_config.init_method,
                                                     config=transformer_config).half().npu()
        output_mc2_open, _ = linear_layer_mc2_open(input_, output_weight_mc2_open)
        loss_mc2_open = torch.mul(output_mc2_open, loss_weight).sum()
        loss_mc2_open.backward()
        output_weight_grad_mc2_open = output_weight_mc2_open.grad

        # result compare
        assert torch.allclose(output_mc2_close, output_mc2_open, rtol=0.005, atol=0.005)
        assert torch.allclose(output_weight_grad_mc2_close, output_weight_grad_mc2_open, rtol=0.005, atol=0.005)
        assert torch.allclose(linear_layer_mc2_close.bias.grad, linear_layer_mc2_open.bias.grad, rtol=0.005, atol=0.005)

    def test_mcore_mc2_column_parallel_linear_frozen(self):
        args = parse_args(None, True)
        args = set_mc2_args(args)
        set_args(args)

        transformer_config = TransformerConfig(num_layers=1,
                                               hidden_size=12,
                                               num_attention_heads=4,
                                               use_cpu_initialization=True)
        transformer_config.sequence_parallel = args.sequence_parallel
        input_size = args.input_size_coeff * args.tensor_model_parallel_size
        output_size = args.vocab_size
        input_ = torch.rand(args.seq_len, args.batch_size, input_size).half().npu()

        # get output_weight
        linear_layer = ColumnParallelLinear(input_size,
                                            output_size,
                                            keep_master_weight_for_test=True,
                                            init_method=transformer_config.init_method,
                                            config=transformer_config).half().npu()
        linear_layer.weight.requires_grad_(False)
        setattr(linear_layer.weight, 'main_grad', linear_layer.weight.clone())
        output_weight_mc2_close = linear_layer.weight
        output_weight_mc2_open = linear_layer.weight

        # close mc2 forward and backward
        set_random_seed(args.seed)
        linear_layer_mc2_close = ColumnParallelLinear(input_size,
                                                      output_size,
                                                      keep_master_weight_for_test=True,
                                                      init_method=transformer_config.init_method,
                                                      config=transformer_config).half().npu()
        linear_layer_mc2_close.weight.requires_grad_(False)
        output_mc2_close, _ = linear_layer_mc2_close(input_, output_weight_mc2_close)

        # open mc2 forward and backward
        initialize_cfg_from_args(args)
        set_random_seed(args.seed)
        linear_layer_mc2_open = ColumnParallelLinear(input_size,
                                                     output_size,
                                                     keep_master_weight_for_test=True,
                                                     init_method=transformer_config.init_method,
                                                     config=transformer_config).half().npu()
        output_mc2_open, _ = linear_layer_mc2_open(input_, output_weight_mc2_open)

        # result compare
        assert torch.allclose(output_mc2_close, output_mc2_open, rtol=0.005, atol=0.005)
