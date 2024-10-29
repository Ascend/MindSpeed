# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import types
from functools import partial
from unittest import mock

import pytest
import torch
from mindspeed import megatron_adaptor

from megatron.training.arguments import parse_args
from megatron.core import parallel_state
from megatron.training.global_vars import set_args
from megatron.core.dist_checkpointing.dict_utils import nested_values, diff
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import DistributedOptimizer, OptimizerConfig, \
    get_megatron_optimizer
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.training.training import get_model
from megatron.training.utils import unwrap_model
from megatron.core.parallel_state import (get_data_parallel_group,
                                          get_data_parallel_group_gloo,
                                          get_data_modulo_expert_parallel_group,
                                          get_data_modulo_expert_parallel_group_gloo)

from mindspeed.utils import _scatter_hccl, _gather_hccl, check_param_hashes_across_dp_replicas_hccl
from mindspeed.optimizer.distrib_optimizer import get_parameter_state_dp_zero_hccl, \
    load_parameter_state_from_dp_zero_hccl
from mindspeed.core.parallel_state import (initialize_model_parallel,
                                           get_data_parallel_group_gloo_replace,
                                           get_data_modulo_expert_parallel_group_gloo_replace)

from unit_tests.common import DistributedTest
from commons import initialize_model_parallel as initialize_model_parallel_comm


def initialize_gpt_model(pre_process=True, post_process=True, seed=0, **config_kwargs):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    # default_config_kwargs=dict(num_layers=8, hidden_size=16, num_attention_heads=8, use_cpu_initialization=True)
    default_config_kwargs = dict(num_layers=1, hidden_size=32, num_attention_heads=8, use_cpu_initialization=True)
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs)
    # pre_process = parallel_state.is_pipeline_first_stage()
    # post_process = parallel_state.is_pipeline_last_stage()
    model = GPTModel(config=transformer_config, transformer_layer_spec=get_gpt_layer_local_spec(), vocab_size=128,
                     max_sequence_length=4,
                     pre_process=pre_process, post_process=post_process)

    model.bfloat16()
    with torch.no_grad():
        for p in model.parameters():
            p.random_()
    return model


def init_mock_args(args, bf16=True):
    args.data_parallel_random_init = False
    args.virtual_pipeline_model_parallel_size = None
    args.bf16 = bf16
    args.accumulate_allreduce_grads_in_fp32 = False
    args.overlap_grad_reduce = False
    args.use_distributed_optimizer = True
    args.ddp_bucket_size = None
    return args


def setup_model_and_optimizer(seed, bf16=True):
    with mock.patch('megatron.training.training.get_args', data_parallel_random_init=False) as mock_args:
        init_mock_args(mock_args.return_value, bf16)
        model = get_model(partial(initialize_gpt_model, seed=seed))

    config = OptimizerConfig(lr=1e-4, bf16=bf16, params_dtype=torch.bfloat16 if bf16 else torch.float,
                             use_distributed_optimizer=bf16)
    optimizer = get_megatron_optimizer(config, model)

    torch.manual_seed(seed + 1)
    model_parallel_cuda_manual_seed(seed + 1)

    for group in optimizer.optimizer.param_groups:
        for p in group['params']:
            if len(optimizer.optimizer.state[p]) == 0:
                optimizer.optimizer.state[p]['exp_avg'] = torch.rand_like(p.data)
                optimizer.optimizer.state[p]['exp_avg_sq'] = torch.rand_like(p.data)

    optimizer.reload_model_params()

    return unwrap_model(model), optimizer


def replace_megatron_function():
    parallel_state.initialize_model_parallel = initialize_model_parallel
    parallel_state.get_data_parallel_group_gloo = get_data_parallel_group_gloo_replace
    parallel_state.get_data_modulo_expert_parallel_group_gloo = get_data_modulo_expert_parallel_group_gloo_replace


def get_parameter_state(disable_gloo_group, optimizer=None):
    if disable_gloo_group:
        if optimizer is None:
            _, optimizer = setup_model_and_optimizer(disable_gloo_group, 2)
        optim_param_state_hccl = optimizer.get_parameter_state_dp_zero()
        return optim_param_state_hccl, optimizer
    else:
        if optimizer is None:
            _, optimizer = setup_model_and_optimizer(disable_gloo_group, 2)
        optim_param_state_gloo = optimizer.get_parameter_state_dp_zero()
        return optim_param_state_gloo, optimizer


def is_tensor_lists_equal(list_a, list_b):
    # First, check if the lengths of the lists are equal.
    if len(list_a) != len(list_b):
        return False

    # Use a loop to compare each tensor one by one.
    for tensor_a, tensor_b in zip(list_a, list_b):
        if not torch.equal(tensor_a, tensor_b):
            return False

    return True


class TestGatherAndScatter(DistributedTest):
    world_size = 8

    @pytest.mark.parametrize("tp_pp", [(2, 1)])
    def test_scatter(self, tp_pp):
        args = parse_args(None, True)
        set_args(args)
        initialize_model_parallel_comm(*tp_pp)

        data_parallel_group = get_data_parallel_group()
        data_parallel_group_gloo = get_data_parallel_group_gloo()
        data_parallel_world_size = data_parallel_group.size()
        data_parallel_rank = torch.distributed.get_rank(data_parallel_group)
        data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
            data_parallel_group
        )

        recv_shape = 10_485_760 * 12 + 20_010

        # GLOO Communication
        recv_tensor_gloo = torch.empty(recv_shape, dtype=torch.float32, device="cpu")
        if data_parallel_rank == 0:
            send_tensors = [torch.rand(recv_shape, dtype=torch.float32, device="cpu") for _
                            in range(data_parallel_world_size)]
        else:
            send_tensors = None
        torch.distributed.scatter(
            recv_tensor_gloo,
            send_tensors,
            data_parallel_global_ranks[0],
            data_parallel_group_gloo,
        )

        # HCCL Slice Communication Optimization
        recv_tensor_hccl = torch.empty(recv_shape, dtype=torch.float32, device="cpu")
        _scatter_hccl(
            recv_tensor_hccl,
            send_tensors,
            data_parallel_global_ranks[0],
            data_parallel_group)

        is_equal = is_tensor_lists_equal([recv_tensor_gloo], [recv_tensor_hccl])
        assert is_equal

    @pytest.mark.parametrize("tp_pp", [(2, 1)])
    def test_gather(self, tp_pp):
        args = parse_args(None, True)
        set_args(args)
        initialize_model_parallel_comm(*tp_pp)

        data_parallel_group = get_data_parallel_group()
        data_parallel_group_gloo = get_data_parallel_group_gloo()
        data_parallel_world_size = data_parallel_group.size()
        data_parallel_rank = torch.distributed.get_rank(data_parallel_group)
        data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
            data_parallel_group
        )

        send_shape = 10_485_760 * 12 + 20_010

        # GLOO Communication
        send_tensor = torch.rand(send_shape, dtype=torch.float32, device="cpu")
        if data_parallel_rank == 0:
            recv_tensors_gloo = [torch.empty(send_shape, dtype=torch.float32, device="cpu") for _
                                 in range(data_parallel_world_size)]
        else:
            recv_tensors_gloo = None
        torch.distributed.gather(
            send_tensor,
            recv_tensors_gloo,
            data_parallel_global_ranks[0],
            data_parallel_group_gloo,
        )

        # HCCL Slice Communication Optimization
        recv_tensors_hccl = [torch.empty(send_shape, dtype=torch.float32) for _
                             in range(data_parallel_world_size)]
        _gather_hccl(
            send_tensor,
            recv_tensors_hccl,
            data_parallel_group,
        )

        if data_parallel_rank == 0:
            is_equal = is_tensor_lists_equal(recv_tensors_gloo, recv_tensors_hccl)
            assert is_equal


class TestCheckParamHash(DistributedTest):
    world_size = 8

    def test_check_param_hashes_across_dp_replicas(self):
        # Setup.
        args = parse_args(None, True)
        set_args(args)
        initialize_model_parallel_comm()
        model = torch.nn.Linear(10000, 100000, bias=False)

        data_parallel_group = get_data_parallel_group()
        data_parallel_rank = torch.distributed.get_rank(data_parallel_group)

        # First check case where all replicas agree.
        model.weight.data.fill_(1.0)
        assert check_param_hashes_across_dp_replicas_hccl([model])

        # Now check case where replica 0 disagrees with all other replicas.
        if data_parallel_rank == 0:
            model.weight.data.fill_(0.0)
        param_hashes_match = check_param_hashes_across_dp_replicas_hccl([model])
        expected_param_hashes_match = (data_parallel_rank == 0)
        assert param_hashes_match == expected_param_hashes_match


class TestDistributedOptimizer(DistributedTest):
    world_size = 8

    @pytest.mark.parametrize("tp_pp", [(2, 1)])
    def test_full_dp_sharding(self, tp_pp):
        args = parse_args(None, True)
        set_args(args)

        # test get_parameter_state()
        args.data_parallel_random_init = False
        initialize_model_parallel_comm(*tp_pp)
        optim_param_state_old, optimizer = get_parameter_state(args.disable_gloo_group)

        args.disable_gloo_group = True
        replace_megatron_function()
        if args.reuse_fp32_param:
            optimizer.get_parameter_state_dp_zero_func = types.MethodType(get_parameter_state_dp_zero_hccl, optimizer)
            optimizer.load_parameter_state_from_dp_zero_func = types.MethodType(load_parameter_state_from_dp_zero_hccl,
                                                                                optimizer)
        else:
            optimizer.get_parameter_state_dp_zero = types.MethodType(get_parameter_state_dp_zero_hccl, optimizer)
            optimizer.load_parameter_state_from_dp_zero = types.MethodType(load_parameter_state_from_dp_zero_hccl,
                                                                           optimizer)
        initialize_model_parallel_comm(*tp_pp)
        optim_param_state_new, _ = get_parameter_state(args.disable_gloo_group, optimizer)
        diffs = diff(optim_param_state_new, optim_param_state_old)
        assert not any(map(bool, diffs)), diffs

        # test load_parameter_state_from_dp_zero()
        optimizer.load_parameter_state_from_dp_zero(optim_param_state_new)
        optim_param_state_new_copy, _ = get_parameter_state(args.disable_gloo_group, optimizer)
        diffs = diff(optim_param_state_new, optim_param_state_new_copy)
        assert not any(map(bool, diffs)), diffs