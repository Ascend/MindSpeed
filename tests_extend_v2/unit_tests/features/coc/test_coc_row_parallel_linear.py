import pytest
import torch
import torch_npu
from mindspeed import megatron_adaptor_v2
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from megatron.core.transformer.transformer_config import TransformerConfig
from mindspeed.core.tensor_parallel.coc_feature.adaptor import MindSpeedCoCRowParallelLinear
from tests_extend.unit_tests.common import DistributedTest
from tests_extend.commons import initialize_model_parallel
from tests_extend.commons import set_random_seed


def set_coc_args(args):
    args.tensor_model_parallel_size = 8
    args.use_unpad = False
    args.seed = 2024
    args.seq_len = 256
    args.input_size_coeff = 256
    args.output_size_coeff = 256
    args.batch_size = 8
    args.optimize_recomp_communication_level = True
    args.sequence_parallel = 1
    args.use_cp_send_recv_overlap = False
    return args


class TestCoC(DistributedTest):
    world_size = 8

    def test_coc_row_parallel_linear_sp(self):
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
        linear_layer = MindSpeedCoCRowParallelLinear(input_size,
                                                     output_size,
                                                     keep_master_weight_for_test=True,
                                                     bias=True, input_is_parallel=True,
                                                     skip_bias_add=False,
                                                     init_method=transformer_config.init_method,
                                                     config=transformer_config).half().npu()
        setattr(linear_layer.weight, 'main_grad', linear_layer.weight.clone())
        loss_weight = torch.rand([args.seq_len, output_size]).half().npu()
        input_ = torch.rand(args.batch_size, args.seq_len, args.input_size_coeff, requires_grad=True)
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
        torch.distributed.reduce_scatter(tensor, scatter_list)
        assert torch.allclose(tensor, output[0], rtol=0.005, atol=0.005)
        x = torch.chunk(input_, chunks=self.world_size, dim=0)[rank]
        torch.distributed.all_reduce(x)
        dLdA = torch.matmul(dLdY.t(), x).sum(dim=0)
        ones = torch.ones(args.seq_len, args.batch_size // self.world_size).half().npu()
        dLdb = torch.matmul(ones.t(), dLdY).sum(dim=0).view(-1)
        assert torch.allclose(dLdA, linear_layer.weight.main_grad, rtol=0.005, atol=0.005)
        assert torch.allclose(dLdb, linear_layer.bias.grad, rtol=0.005, atol=0.005)
    
    def test_coc_row_parallel_linear_sp_fused_kernel(self):
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
        linear_layer = MindSpeedCoCRowParallelLinear(input_size,
                                                     output_size,
                                                     keep_master_weight_for_test=True,
                                                     bias=True, input_is_parallel=True,
                                                     skip_bias_add=False,
                                                     init_method=transformer_config.init_method,
                                                     config=transformer_config).half().npu()
        setattr(linear_layer.weight, 'main_grad', linear_layer.weight.clone())
        loss_weight = torch.rand([args.seq_len, output_size]).half().npu()
        input_ = torch.rand(args.batch_size, args.seq_len, args.input_size_coeff, requires_grad=True)
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
        torch.distributed.reduce_scatter(tensor, scatter_list)
        assert torch.allclose(tensor, output[0], rtol=0.005, atol=0.005)
        x = torch.chunk(input_, chunks=self.world_size, dim=0)[rank]
        torch.distributed.all_reduce(x)
        dLdA = torch.matmul(dLdY.t(), x).sum(dim=0)
        ones = torch.ones(args.seq_len, args.batch_size // self.world_size).half().npu()
        dLdb = torch.matmul(ones.t(), dLdY).sum(dim=0).view(-1)
        assert torch.allclose(dLdA, linear_layer.weight.grad, rtol=0.005, atol=0.005)
        assert torch.allclose(dLdb, linear_layer.bias.grad, rtol=0.005, atol=0.005)

    def test_coc_row_parallel_linear(self):
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
        linear_layer = MindSpeedCoCRowParallelLinear(input_size,
                                                     output_size,
                                                     keep_master_weight_for_test=True,
                                                     bias=True, input_is_parallel=True,
                                                     skip_bias_add=False,
                                                     init_method=transformer_config.init_method,
                                                     config=transformer_config).half().npu()
        setattr(linear_layer.weight, 'main_grad', linear_layer.weight.clone())
        loss_weight = torch.rand([args.seq_len, output_size]).half().npu()
        input_ = torch.rand(args.batch_size, args.seq_len, args.input_size_coeff, requires_grad=True)
        input_ = input_.half().npu()
        output = linear_layer(input_)
        loss = torch.mul(output[0], loss_weight).sum()
        # Backward
        loss.backward()
        dLdY = loss_weight
        rank = torch.distributed.get_rank()
        res = torch.matmul(input_, linear_layer.weight.npu().T)
        torch.distributed.all_reduce(res)
        assert torch.allclose(res, output[0], rtol=0.005, atol=0.005)
        x = torch.chunk(input_, chunks=self.world_size, dim=0)[rank]
        torch.distributed.all_reduce(x)
        dLdA = torch.matmul(dLdY.t(), x).sum(dim=0)
        ones = torch.ones(args.seq_len, args.batch_size).half().npu()
        dLdb = torch.matmul(ones.t(), dLdY).sum(dim=0).view(-1)
        assert torch.allclose(dLdA, linear_layer.weight.grad, rtol=0.005, atol=0.005)
        assert torch.allclose(dLdb, linear_layer.bias.grad, rtol=0.005, atol=0.005)

    def test_coc_row_parallel_linear_fused_kernel(self):
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
        linear_layer = MindSpeedCoCRowParallelLinear(input_size,
                                                     output_size,
                                                     keep_master_weight_for_test=True,
                                                     bias=True, input_is_parallel=True,
                                                     skip_bias_add=False,
                                                     init_method=transformer_config.init_method,
                                                     config=transformer_config).half().npu()
        setattr(linear_layer.weight, 'main_grad', linear_layer.weight.clone())
        loss_weight = torch.rand([args.seq_len, output_size]).half().npu()
        input_ = torch.rand(args.batch_size, args.seq_len, args.input_size_coeff, requires_grad=True)
        input_ = input_.half().npu()
        output = linear_layer(input_)
        loss = torch.mul(output[0], loss_weight).sum()
        # Backward
        loss.backward()
        dLdY = loss_weight
        rank = torch.distributed.get_rank()
        res = torch.matmul(input_, linear_layer.weight.npu().T)
        torch.distributed.all_reduce(res)
        assert torch.allclose(res, output[0], rtol=0.005, atol=0.005) 
        x = torch.chunk(input_, chunks=self.world_size, dim=0)[rank]
        torch.distributed.all_reduce(x)
        dLdA = torch.matmul(dLdY.t(), x).sum(dim=0)
        ones = torch.ones(args.seq_len, args.batch_size).half().npu()
        dLdb = torch.matmul(ones.t(), dLdY).sum(dim=0).view(-1)
        assert torch.allclose(dLdA, linear_layer.weight.grad, rtol=0.005, atol=0.005)
        assert torch.allclose(dLdb, linear_layer.bias.grad, rtol=0.005, atol=0.005)
