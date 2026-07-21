# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from functools import partial
import copy
import itertools

import pytest
import torch

from mindspeed import megatron_adaptor  # noqa: F401
from mindspeed.megatron_adaptor import repatch
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.timers import DummyTimer
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.training.training import get_model
from megatron.training.utils import unwrap_model

from tests_extend.unit_tests.common import DistributedTest
from tests_extend.commons import set_random_seed, initialize_model_parallel


def initialize_gpt_model(pre_process=True, post_process=True, seed=0, **config_kwargs):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    default_config_kwargs = dict(num_layers=8, hidden_size=512, num_attention_heads=32, use_cpu_initialization=True)
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs)
    model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=1024,
        max_sequence_length=64,
        pre_process=pre_process,
        post_process=post_process,
    )

    model.bfloat16()
    with torch.no_grad():
        for p in model.parameters():
            p.random_()
    return model


def init_mock_args(
    args, use_distributed_optimizer=False, swap_optimizer=False, swap_optimizer_times=16, optimizer='adam'
):
    args.data_parallel_random_init = False
    args.virtual_pipeline_model_parallel_size = None
    args.bf16 = True
    args.accumulate_allreduce_grads_in_fp32 = True
    args.use_distributed_optimizer = use_distributed_optimizer
    args.ddp_bucket_size = None
    args.swap_optimizer = swap_optimizer
    args.swap_optimizer_times = swap_optimizer_times
    args.optimizer = optimizer
    args.num_query_groups = None
    # Muon optimizer requires use_layer_wise_distributed_optimizer=True.
    # Normally set by MuonOptimizerFeature.post_validate_args, but repatch()
    # does not call post_validate_features_args, so we set it here explicitly.
    if optimizer == 'muon' and use_distributed_optimizer:
        args.use_layer_wise_distributed_optimizer = True
        args.use_distributed_optimizer = False
    return args


def setup_model_and_optimizer(seed, use_distributed_optimizer=False):
    model = get_model(partial(initialize_gpt_model, seed=seed, bf16=True))
    set_random_seed(seed)
    config = OptimizerConfig(
        lr=1e-4, bf16=True, params_dtype=torch.bfloat16, use_distributed_optimizer=use_distributed_optimizer
    )
    config.timers = Timers()
    optimizer = get_megatron_optimizer(config, model)

    for group in optimizer.optimizer.param_groups:
        for p in group['params']:
            if len(optimizer.optimizer.state[p]) == 0:
                optimizer.optimizer.state[p]['exp_avg'] = torch.rand_like(p.data)
                optimizer.optimizer.state[p]['exp_avg_sq'] = torch.rand_like(p.data)
    optimizer.reload_model_params()
    return unwrap_model(model), optimizer


def setup_model_and_muon_optimizer(seed):
    model = get_model(partial(initialize_gpt_model, seed=seed, bf16=True))
    set_random_seed(seed)
    config = OptimizerConfig(
        optimizer='muon',
        lr=1e-4,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
    )
    config.timers = Timers()
    optimizer = get_megatron_optimizer(config, model)

    # Initialize optimizer states for Muon (momentum_buffer) and Adam (exp_avg, exp_avg_sq)
    for sub_opt in optimizer.chained_optimizers:
        for group in sub_opt.optimizer.param_groups:
            for p in group['params']:
                if len(sub_opt.optimizer.state[p]) == 0:
                    if hasattr(sub_opt.optimizer, 'orthogonalize'):
                        sub_opt.optimizer.state[p]['momentum_buffer'] = torch.rand_like(p.data)
                    else:
                        sub_opt.optimizer.state[p]['exp_avg'] = torch.rand_like(p.data)
                        sub_opt.optimizer.state[p]['exp_avg_sq'] = torch.rand_like(p.data)
    optimizer.reload_model_params()
    return unwrap_model(model), optimizer


def reset_swap_distributed_optimizer():
    """Reset SwapDistributedOptimizer class-level mutable state."""
    from mindspeed.core.optimizer.swap_optimizer.swap_optimizer import SwapDistributedOptimizer

    SwapDistributedOptimizer.swap_to_device_stream = None
    SwapDistributedOptimizer.swap_to_host_stream = None
    SwapDistributedOptimizer.swap_to_device_events_map = {}
    SwapDistributedOptimizer.swap_to_host_events_map = {}
    SwapDistributedOptimizer.copy_to_model_param_events_map = {}
    SwapDistributedOptimizer.param_to_cpu_states_map = {}
    SwapDistributedOptimizer.param_to_device_states_map = {}
    SwapDistributedOptimizer.main_param_to_model_param_map = {}
    SwapDistributedOptimizer.no_swap_params = set()
    SwapDistributedOptimizer.step_count = 0
    SwapDistributedOptimizer.ALL_OPTIMIZER = []


def reset_swap_optimizer_mixin():
    """Reset SwapOptimizerMixin class-level mutable state."""
    from mindspeed.core.optimizer.swap_muon.swap_muon import SwapOptimizerMixin

    SwapOptimizerMixin._swap_to_device_stream = None
    SwapOptimizerMixin._swap_to_host_stream = None
    SwapOptimizerMixin._swap_numel = 0
    SwapOptimizerMixin._param_to_cpu_states = {}
    SwapOptimizerMixin._state_map = {}
    SwapOptimizerMixin._swap_to_device_events = {}
    SwapOptimizerMixin._swap_to_host_events = {}
    SwapOptimizerMixin._copy_to_model_events = {}
    SwapOptimizerMixin._main_param_to_model_param = {}
    SwapOptimizerMixin._step_count = 0
    SwapOptimizerMixin._total_optimizer_count = 0


class Timers:
    def __init__(self, *args, **kwargs):
        self._dummy_timer = DummyTimer()

    def __call__(self, *args, **kwargs):
        return self._dummy_timer


class TestDistributedOptimizer(DistributedTest):
    world_size = 8

    @pytest.mark.parametrize("is_deterministic", [False])
    @pytest.mark.parametrize("overlap_grad_reduce", [pytest.param(True, marks=pytest.mark.slow), False])
    @pytest.mark.parametrize("overlap_param_gather", [pytest.param(True, marks=pytest.mark.slow), False])
    @pytest.mark.parametrize(
        "tp_pp",
        [
            pytest.param((4, 1), marks=pytest.mark.slow),
            (2, 2),
            pytest.param((8, 1), marks=pytest.mark.slow),
        ],
    )
    def test_swap_optimizer(self, tp_pp, is_deterministic, overlap_grad_reduce, overlap_param_gather):
        args = parse_args(None, True)
        args.npu_deterministic = is_deterministic
        args.overlap_grad_reduce = overlap_grad_reduce
        args.overlap_param_gather = overlap_param_gather
        set_args(args)

        # truth
        init_mock_args(args, use_distributed_optimizer=True)
        initialize_model_parallel(tensor_model_parallel_size=tp_pp[0], pipeline_model_parallel_size=tp_pp[1])
        _, optimizer = setup_model_and_optimizer(seed=5, use_distributed_optimizer=True)
        for _ in range(10):
            for float16_group in optimizer.chained_optimizers[0].model_float16_groups:
                for p in float16_group:
                    p.grad = torch.randn_like(p.data, dtype=p.data.dtype)
            optimizer.step()
            if overlap_param_gather:
                for model_chunk in optimizer.model_chunks:
                    model_chunk.start_param_sync(force_sync=True)
                torch.cuda.synchronize()
        truth_params = copy.deepcopy(list(itertools.chain(*optimizer.chained_optimizers[0].model_float16_groups)))

        # swap_optimizer
        init_mock_args(args, use_distributed_optimizer=True, swap_optimizer=True)
        initialize_model_parallel(tensor_model_parallel_size=tp_pp[0], pipeline_model_parallel_size=tp_pp[1])
        _, optimizer = setup_model_and_optimizer(seed=5, use_distributed_optimizer=True)
        for _ in range(10):
            for float16_group in optimizer.chained_optimizers[0].model_float16_groups:
                for p in float16_group:
                    p.grad = torch.randn_like(p.data, dtype=p.data.dtype)
            optimizer.step()
            if overlap_param_gather:
                for model_chunk in optimizer.model_chunks:
                    model_chunk.start_param_sync(force_sync=True)
                torch.cuda.synchronize()
        swap_optimizer_params = copy.deepcopy(
            list(itertools.chain(*optimizer.chained_optimizers[0].model_float16_groups))
        )

        for p, swap_optimizer_p in zip(truth_params, swap_optimizer_params):
            if is_deterministic:
                assert torch.allclose(p.data, swap_optimizer_p.data, rtol=0, atol=0)
            else:
                assert torch.allclose(p.data, swap_optimizer_p.data, rtol=0.005, atol=0.005)

    def test_swap_optimizer_deferred_release(self):
        """Verify swap_optimizer_times=0 (deferred release) produces the same
        results as swap_optimizer_times=16 (default mode).
        """
        from mindspeed.core.optimizer.swap_optimizer.swap_optimizer import SwapDistributedOptimizer

        tp_pp = (2, 2)
        args = parse_args(None, True)
        args.npu_deterministic = False
        args.overlap_grad_reduce = False
        args.overlap_param_gather = False
        set_args(args)

        # Baseline: swap_optimizer with times=16 (default, no deferred release)
        reset_swap_distributed_optimizer()
        init_mock_args(args, use_distributed_optimizer=True, swap_optimizer=True, swap_optimizer_times=16)
        repatch(vars(args))
        initialize_model_parallel(tensor_model_parallel_size=tp_pp[0], pipeline_model_parallel_size=tp_pp[1])
        _, optimizer = setup_model_and_optimizer(seed=5, use_distributed_optimizer=True)
        for _ in range(10):
            for float16_group in optimizer.chained_optimizers[0].model_float16_groups:
                for p in float16_group:
                    p.grad = torch.randn_like(p.data, dtype=p.data.dtype)
            optimizer.step()
            torch.cuda.synchronize()
        baseline_params = copy.deepcopy(list(itertools.chain(*optimizer.chained_optimizers[0].model_float16_groups)))

        # Deferred release: swap_optimizer with times=0
        reset_swap_distributed_optimizer()
        init_mock_args(args, use_distributed_optimizer=True, swap_optimizer=True, swap_optimizer_times=0)
        repatch(vars(args))
        initialize_model_parallel(tensor_model_parallel_size=tp_pp[0], pipeline_model_parallel_size=tp_pp[1])
        _, optimizer = setup_model_and_optimizer(seed=5, use_distributed_optimizer=True)
        for _ in range(10):
            for float16_group in optimizer.chained_optimizers[0].model_float16_groups:
                for p in float16_group:
                    p.grad = torch.randn_like(p.data, dtype=p.data.dtype)
            optimizer.step()
            torch.cuda.synchronize()
        deferred_params = copy.deepcopy(list(itertools.chain(*optimizer.chained_optimizers[0].model_float16_groups)))

        # Verify numerical consistency
        for p, dp in zip(baseline_params, deferred_params):
            assert torch.allclose(p.data, dp.data, rtol=0.005, atol=0.005)

        # Verify class state is properly reset after each iteration
        assert SwapDistributedOptimizer.step_count == 0

    def test_swap_muon_deferred_release(self):
        """Verify swap_optimizer_times=0 (deferred release) produces the same
        results as swap_optimizer_times=16 (default mode) for Muon optimizer.
        """
        from mindspeed.core.optimizer.swap_muon.swap_muon import SwapOptimizerMixin

        tp_pp = (2, 2)
        args = parse_args(None, True)
        args.npu_deterministic = False
        args.overlap_grad_reduce = False
        args.overlap_param_gather = False
        set_args(args)

        # Repatch with muon optimizer so that get_megatron_optimizer recognizes it
        init_mock_args(
            args, use_distributed_optimizer=True, swap_optimizer=True, swap_optimizer_times=16, optimizer='muon'
        )
        repatch(vars(args))

        # Baseline: swap_optimizer with times=16 (default, no deferred release)
        reset_swap_optimizer_mixin()
        initialize_model_parallel(tensor_model_parallel_size=tp_pp[0], pipeline_model_parallel_size=tp_pp[1])
        _, optimizer = setup_model_and_muon_optimizer(seed=5)
        for _ in range(10):
            for sub_opt in optimizer.chained_optimizers:
                for float16_group in sub_opt.float16_groups:
                    for p in float16_group:
                        p.grad = torch.randn_like(p.data, dtype=p.data.dtype)
            optimizer.step()
            torch.cuda.synchronize()
        baseline_params = copy.deepcopy(list(itertools.chain(*optimizer.chained_optimizers[0].float16_groups)))

        # Deferred release: swap_optimizer with times=0
        init_mock_args(
            args, use_distributed_optimizer=True, swap_optimizer=True, swap_optimizer_times=0, optimizer='muon'
        )
        reset_swap_optimizer_mixin()
        initialize_model_parallel(tensor_model_parallel_size=tp_pp[0], pipeline_model_parallel_size=tp_pp[1])
        _, optimizer = setup_model_and_muon_optimizer(seed=5)
        for _ in range(10):
            for sub_opt in optimizer.chained_optimizers:
                for float16_group in sub_opt.float16_groups:
                    for p in float16_group:
                        p.grad = torch.randn_like(p.data, dtype=p.data.dtype)
            optimizer.step()
            torch.cuda.synchronize()
        deferred_params = copy.deepcopy(list(itertools.chain(*optimizer.chained_optimizers[0].float16_groups)))

        # Verify numerical consistency
        for p, dp in zip(baseline_params, deferred_params):
            assert torch.allclose(p.data, dp.data, rtol=0.005, atol=0.005)

        # Verify class state is properly reset after each iteration
        assert SwapOptimizerMixin._step_count == 0
