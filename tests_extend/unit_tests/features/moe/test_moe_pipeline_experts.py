import functools
import pytest
import torch

from unit_tests.common import DistributedTest
from commons import set_random_seed

from mindspeed import megatron_adaptor
from mindspeed.moe.experts import Experts
from mindspeed.moe.pipe_experts import PipeExpert
from mindspeed.moe.utils import _AllToAll
from mindspeed.moe.mixtral_parallel_mlpbm import MixtralParallelMLPBM
from mindspeed.core.tensor_parallel.layers import (row_parallel_moe, column_parallel_moe,
                                                   linear_with_grad_accumulation_and_async_allreduce_moe)
from mindspeed.core.distributed.param_and_grad_buffer import pipe_register_grad_ready_wrapper
from mindspeed.patch_utils import MindSpeedPatchesManager as pm

from megatron.core.transformer import TransformerConfig
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer, _ParamAndGradBucketGroup
from megatron.core.tensor_parallel.layers import RowParallelLinear, ColumnParallelLinear
from megatron.legacy.model.transformer import ParallelMLP
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.core.parallel_state import (get_expert_model_parallel_group, destroy_model_parallel,
                                          initialize_model_parallel)
from megatron.core.tensor_parallel.layers import linear_with_grad_accumulation_and_async_allreduce


def get_copied_function(func):
    @functools.wraps(func)
    def copied_function(*args, **kwargs):
        return func(*args, **kwargs)

    return copied_function


ms_row_parallel_forward = get_copied_function(row_parallel_moe)
ms_column_parallel_forward = get_copied_function(column_parallel_moe)
ms_linear_with_grad_accumulation_and_async_allreduce = get_copied_function(
    linear_with_grad_accumulation_and_async_allreduce_moe)
ms_register_grad_ready = pipe_register_grad_ready_wrapper(_ParamAndGradBucketGroup.register_grad_ready)

mg_row_parallel_forward = get_copied_function(RowParallelLinear.forward)
mg_column_parallel_forward = get_copied_function(ColumnParallelLinear.forward)
mg_linear_with_grad_accumulation_and_async_allreduce = get_copied_function(
    linear_with_grad_accumulation_and_async_allreduce)
mg_register_grad_ready = get_copied_function(_ParamAndGradBucketGroup.register_grad_ready)


def switch_patch(use_pipe_experts):
    if use_pipe_experts:
        pm.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear.forward',
                          ms_row_parallel_forward, True)
        pm.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear.forward',
                          ms_column_parallel_forward, True)
        pm.register_patch('megatron.core.tensor_parallel.layers.linear_with_grad_accumulation_and_async_allreduce',
                          ms_linear_with_grad_accumulation_and_async_allreduce, True)
        pm.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.register_grad_ready',
                          ms_register_grad_ready, True)
    else:
        pm.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear.forward',
                          mg_row_parallel_forward, True)
        pm.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear.forward',
                          mg_column_parallel_forward, True)
        pm.register_patch('megatron.core.tensor_parallel.layers.linear_with_grad_accumulation_and_async_allreduce',
                          mg_linear_with_grad_accumulation_and_async_allreduce, True)
        pm.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.register_grad_ready',
                          mg_register_grad_ready, True)
    pm.apply_patches()


class TestMOELayer(DistributedTest):
    world_size = 8
    args = parse_args(None, True)
    mixtral_parallel_mlp = None
    parallel_mlp = None
    input_tensor = None
    rtol = 1e-05
    atol = 1e-05

    def get_experts_output(self, input_tensor, test_args_tuple, expert):
        self.set_test_args(*test_args_tuple)
        set_args(self.args)
        switch_patch(self.args.use_pipe_experts)

        num_local_experts = self.args.num_experts // self.args.expert_model_parallel_size
        experts = Experts(expert, self.args.num_experts).npu()
        ep_group = get_expert_model_parallel_group()
        d_model = input_tensor.shape[-1]

        if self.args.use_pipe_experts:
            output_tensor = PipeExpert.apply(experts, input_tensor, self.args.expert_model_parallel_size,
                                             num_local_experts, self.args.sequence_parallel,
                                             self.args.pipe_experts_multi_data, self.args.pipe_experts_multi_stream)
        else:
            # see also mindspeed.moe.moe_layers
            input_tensor_ = _AllToAll.apply(ep_group, input_tensor)
            input_tensor_ = input_tensor_.reshape(self.args.expert_model_parallel_size, num_local_experts, -1, d_model)
            output_tensor = experts(input_tensor_)
            output_tensor = _AllToAll.apply(ep_group, output_tensor)

        output_tensor = output_tensor.reshape(self.args.expert_model_parallel_size * num_local_experts, -1, d_model)
        output_tensor.sum().backward()

        return output_tensor, input_tensor.grad

    def init_args(self):
        self.args.seed = 2024

        self.args.num_layers = 2
        self.args.seq_len = 128
        self.args.hidden_size = 8
        self.args.num_attention_heads = 8
        self.args.use_cpu_initialization = True
        self.args.use_fp16 = True
        self.args.num_experts = 8
        self.args.ffn_hidden_size = 8
        self.args.pipeline_model_parallel_size = 1

        # use @pytest.mark.parametrize to set the following parameters
        self.args.tensor_model_parallel_size = None
        self.args.expert_model_parallel_size = None
        self.args.sequence_parallel = None
        self.args.use_pipe_experts = None
        self.args.pipe_experts_multi_stream = None
        self.args.pipe_experts_multi_data = None

    def set_test_args(self, sequence_parallel, use_pipe_experts, pipe_experts_multi_stream, pipe_experts_multi_data):
        self.args.sequence_parallel = sequence_parallel
        self.args.use_pipe_experts = use_pipe_experts
        self.args.pipe_experts_multi_stream = pipe_experts_multi_stream
        self.args.pipe_experts_multi_data = pipe_experts_multi_data

    def init_parallel_mlp(self):
        transformer_config = TransformerConfig(
            num_layers=self.args.num_layers,
            hidden_size=self.args.hidden_size,
            num_attention_heads=self.args.num_attention_heads,
            ffn_hidden_size=self.args.hidden_size,
            use_cpu_initialization=self.args.use_cpu_initialization,
            fp16=self.args.use_fp16,
            sequence_parallel=self.args.sequence_parallel,
            tensor_model_parallel_size=self.args.tensor_model_parallel_size,
            expert_model_parallel_size=self.args.expert_model_parallel_size,
            num_moe_experts=self.args.num_experts
        )
        self.mixtral_parallel_mlp = MixtralParallelMLPBM(transformer_config)
        self.parallel_mlp = ParallelMLP(transformer_config)

    def init_model_parallel(self):
        destroy_model_parallel()
        initialize_model_parallel(
            tensor_model_parallel_size=self.args.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.args.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=None,
            pipeline_model_parallel_split_rank=None,
            expert_model_parallel_size=self.args.expert_model_parallel_size
        )

    def init_input_tensor(self, tp_ep):
        tp_size, ep_size = tp_ep
        dim_0_size = tp_size * ep_size
        dim_1_size = self.args.seq_len // dim_0_size
        dim_2_size = self.args.hidden_size
        self.input_tensor = torch.randn((dim_0_size, dim_1_size, dim_2_size), device=torch.npu.current_device(),
                                        dtype=torch.float16)

    def inner_test_moe_layer(self, tp_ep, base_tuple, pipe_tuple):
        # init args and model
        set_random_seed(self.args.seed)
        self.args.tensor_model_parallel_size, self.args.expert_model_parallel_size = tp_ep
        self.args.sequence_parallel = pipe_tuple[0]
        self.args.use_pipe_experts = True
        set_args(self.args)
        self.init_model_parallel()
        self.init_parallel_mlp()

        # mixtral_parallel_mlp
        input_tensor_base = self.input_tensor.clone().detach()
        input_tensor_pipe = self.input_tensor.clone().detach()
        input_tensor_base.requires_grad = True
        input_tensor_pipe.requires_grad = True
        output_base, grad_input_base = self.get_experts_output(input_tensor_base, base_tuple, self.mixtral_parallel_mlp)
        output_pipe, grad_input_pipe = self.get_experts_output(input_tensor_pipe, pipe_tuple, self.mixtral_parallel_mlp)
        assert torch.allclose(output_base, output_pipe, rtol=self.rtol, atol=self.atol)
        assert torch.allclose(grad_input_base, grad_input_pipe, rtol=self.rtol, atol=self.atol)

        # parallel_mlp
        input_tensor_base = self.input_tensor.clone().detach()
        input_tensor_pipe = self.input_tensor.clone().detach()
        input_tensor_base.requires_grad = True
        input_tensor_pipe.requires_grad = True
        output_base, grad_input_base = self.get_experts_output(input_tensor_base, base_tuple, self.parallel_mlp)
        output_pipe, grad_input_pipe = self.get_experts_output(input_tensor_pipe, pipe_tuple, self.parallel_mlp)
        assert torch.allclose(output_base, output_pipe, rtol=self.rtol, atol=self.atol)
        assert torch.allclose(grad_input_base, grad_input_pipe, rtol=self.rtol, atol=self.atol)

    @pytest.mark.parametrize("tp_ep", [(2, 2), (4, 1), (8, 1)])
    @pytest.mark.parametrize("base_tuple", [(True, False, False, 1)])
    @pytest.mark.parametrize("pipe_tuple", [(True, True, False, 1), (True, True, True, 1), (True, True, False, 4),
                                            (True, True, True, 4)])
    def test_mlp_with_sp(self, tp_ep, base_tuple: tuple, pipe_tuple: tuple):
        self.init_args()
        self.init_input_tensor(tp_ep)
        self.inner_test_moe_layer(tp_ep, base_tuple, pipe_tuple)

    @pytest.mark.parametrize("tp_ep", [(2, 1), (1, 2), (1, 1), (1, 8)])
    @pytest.mark.parametrize("base_tuple", [(False, False, False, 1)])
    @pytest.mark.parametrize("pipe_tuple", [(False, True, False, 1), (False, True, False, 4)])
    def test_mlp_without_sp(self, tp_ep, base_tuple: tuple, pipe_tuple: tuple):
        self.init_args()
        self.init_input_tensor(tp_ep)
        self.inner_test_moe_layer(tp_ep, base_tuple, pipe_tuple)
