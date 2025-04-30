import pytest
import torch
import torch_npu
from mindspeed import megatron_adaptor_v2
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from megatron.core.transformer.transformer_config import TransformerConfig
from mindspeed.core.tensor_parallel.coc_feature.adaptor import MindSpeedCoCColumnParallelLinear
from tests_extend.unit_tests.common import DistributedTest
from tests_extend.commons import initialize_model_parallel
from tests_extend.commons import set_random_seed


def set_coc_args(args):
    args.tensor_model_parallel_size = 8
    args.use_unpad = False
    args.seed = 2024
    args.seq_len = 128
    args.input_size_coeff = 128
    args.output_size_coeff = 128
    args.batch_size = 8
    args.optimize_recomp_communication_level = True
    args.sequence_parallel = 1
    args.use_cp_send_recv_overlap = False
    return args


class TestCoC(DistributedTest):
    world_size = 8

    def test_coc_column_parallel_linear_sp(self):
        args = parse_args(None, True)
        args = set_coc_args(args)
        set_args(args)
        transformer_config = TransformerConfig(num_layers=1,
                                               hidden_size=12,
                                               num_attention_heads=4,
                                               use_cpu_initialization=True)

        # ----------COC SETTING----------
        transformer_config.coc_mode = 2
        transformer_config.coc_parallel_num = 2
        transformer_config.coc_fused_kernel = False
        transformer_config.tensor_model_parallel_size = args.tensor_model_parallel_size
        transformer_config.sequence_parallel = 1
        # -------------EMD---------------
    
        initialize_model_parallel(args.tensor_model_parallel_size, 1)
        set_random_seed(args.seed)
        input_size = args.input_size_coeff * args.tensor_model_parallel_size
        output_size = args.output_size_coeff * args.tensor_model_parallel_size
        linear_layer = MindSpeedCoCColumnParallelLinear(input_size,
                                                        output_size,
                                                        keep_master_weight_for_test=True,
                                                        init_method=transformer_config.init_method,
                                                        config=transformer_config).half().npu()
        setattr(linear_layer.weight, 'main_grad', linear_layer.weight.clone())
        loss_weight = torch.rand([args.seq_len, args.output_size_coeff]).half().npu()
        input_ = torch.rand(args.batch_size, args.seq_len, input_size, requires_grad=True).half().npu()
        output = linear_layer(input_)
        gather_list = [torch.zeros(input_.shape).half().npu() for _ in range(self.world_size)]
        torch.distributed.all_gather(gather_list, input_)
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
        assert torch.allclose(dLdA, linear_layer.weight.main_grad, rtol=0.005, atol=0.005)
        assert torch.allclose(dLdb, linear_layer.bias.grad, rtol=0.005, atol=0.005)

    def test_coc_column_parallel_linear_sp_fused_kernel(self):
        args = parse_args(None, True)
        args = set_coc_args(args)
        set_args(args)
        transformer_config = TransformerConfig(num_layers=1,
                                               hidden_size=12,
                                               num_attention_heads=4,
                                               use_cpu_initialization=True)

        # ----------COC SETTING----------
        transformer_config.coc_mode = 2
        transformer_config.coc_parallel_num = 4
        transformer_config.coc_fused_kernel = True
        transformer_config.tensor_model_parallel_size = args.tensor_model_parallel_size
        transformer_config.sequence_parallel = 1
        # -------------EMD---------------
        
        initialize_model_parallel(args.tensor_model_parallel_size, 1)
        set_random_seed(args.seed)
        input_size = args.input_size_coeff * args.tensor_model_parallel_size
        output_size = args.output_size_coeff * args.tensor_model_parallel_size
        linear_layer = MindSpeedCoCColumnParallelLinear(input_size,
                                                        output_size,
                                                        keep_master_weight_for_test=True,
                                                        init_method=transformer_config.init_method,
                                                        config=transformer_config).half().npu()
        setattr(linear_layer.weight, 'main_grad', linear_layer.weight.clone())
        loss_weight = torch.rand([args.seq_len, args.output_size_coeff]).half().npu()
        input_ = torch.rand(args.batch_size, args.seq_len, input_size, requires_grad=True).half().npu()
        output = linear_layer(input_)
        gather_list = [torch.zeros(input_.shape).half().npu() for _ in range(self.world_size)]
        torch.distributed.all_gather(gather_list, input_)
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
    
    def test_coc_column_parallel_linear(self):
        args = parse_args(None, True)
        args = set_coc_args(args)
        set_args(args)
        transformer_config = TransformerConfig(num_layers=1,
                                               hidden_size=12,
                                               num_attention_heads=4,
                                               use_cpu_initialization=True)
        
        # ----------COC SETTING----------
        transformer_config.coc_mode = 2
        transformer_config.coc_parallel_num = 8
        transformer_config.coc_fused_kernel = False
        transformer_config.tensor_model_parallel_size = args.tensor_model_parallel_size
        transformer_config.sequence_parallel = 0
        # -------------EMD---------------
        
        initialize_model_parallel(args.tensor_model_parallel_size, 1)
        set_random_seed(args.seed)
        input_size = args.input_size_coeff * args.tensor_model_parallel_size
        output_size = args.output_size_coeff * args.tensor_model_parallel_size
        linear_layer = MindSpeedCoCColumnParallelLinear(input_size,
                                                        output_size,
                                                        keep_master_weight_for_test=True,
                                                        init_method=transformer_config.init_method,
                                                        config=transformer_config).half().npu()
        setattr(linear_layer.weight, 'main_grad', linear_layer.weight.clone())
        loss_weight = torch.rand([args.seq_len, args.output_size_coeff]).half().npu()
        input_ = torch.rand(args.batch_size, args.seq_len, input_size, requires_grad=True).half().npu()
        output = linear_layer(input_)

        output_naive = torch.matmul(input_, linear_layer.weight.t())
        assert torch.allclose(output_naive, output[0], rtol=0.005, atol=0.005)
        loss = torch.mul(output[0], loss_weight).sum()
        # Backward
        loss.backward()
        dLdY = loss_weight
        dLdA = torch.matmul(dLdY.t(), input_).sum(dim=0)
        ones = torch.ones(args.seq_len, args.batch_size * 1).half().npu()
        dLdb = torch.matmul(ones.t(), dLdY).sum(dim=0).view(-1)
        assert torch.allclose(dLdA, linear_layer.weight.grad, rtol=0.005, atol=0.005)
        assert torch.allclose(dLdb, linear_layer.bias.grad, rtol=0.005, atol=0.005)

    def test_coc_column_parallel_linear_fused_kernel(self):
        args = parse_args(None, True)
        args = set_coc_args(args)
        set_args(args)
        transformer_config = TransformerConfig(num_layers=1,
                                               hidden_size=12,
                                               num_attention_heads=4,
                                               use_cpu_initialization=True)
        
        # ----------COC SETTING----------
        transformer_config.coc_mode = 2
        transformer_config.coc_parallel_num = 1
        transformer_config.coc_fused_kernel = True
        transformer_config.tensor_model_parallel_size = args.tensor_model_parallel_size
        transformer_config.sequence_parallel = 0
        # -------------EMD---------------
        
        initialize_model_parallel(args.tensor_model_parallel_size, 1)
        set_random_seed(args.seed)
        input_size = args.input_size_coeff * args.tensor_model_parallel_size
        output_size = args.output_size_coeff * args.tensor_model_parallel_size
        linear_layer = MindSpeedCoCColumnParallelLinear(input_size,
                                                        output_size,
                                                        keep_master_weight_for_test=True,
                                                        init_method=transformer_config.init_method,
                                                        config=transformer_config).half().npu()
        setattr(linear_layer.weight, 'main_grad', linear_layer.weight.clone())
        loss_weight = torch.rand([args.seq_len, args.output_size_coeff]).half().npu()
        input_ = torch.rand(args.batch_size, args.seq_len, input_size, requires_grad=True).half().npu()
        output = linear_layer(input_)
        output_naive = torch.matmul(input_, linear_layer.weight.t())
        assert torch.allclose(output_naive, output[0], rtol=0.005, atol=0.005)
        loss = torch.mul(output[0], loss_weight).sum()
        # Backward
        loss.backward()
        dLdY = loss_weight
        dLdA = torch.matmul(dLdY.t(), input_).sum(dim=0)
        ones = torch.ones(args.seq_len, args.batch_size * 1).half().npu()
        dLdb = torch.matmul(ones.t(), dLdY).sum(dim=0).view(-1)
        assert torch.allclose(dLdA, linear_layer.weight.grad, rtol=0.005, atol=0.005)
        assert torch.allclose(dLdb, linear_layer.bias.grad, rtol=0.005, atol=0.005)