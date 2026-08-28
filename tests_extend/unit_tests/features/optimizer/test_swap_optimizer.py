# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from functools import partial
import itertools
import os

import pytest
import torch
import torch.distributed as dist

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


FAST_OPTIMIZER_STEPS = 2


def initialize_gpt_model(pre_process=True, post_process=True, seed=0, **config_kwargs):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    default_config_kwargs = dict(num_layers=2, hidden_size=128, num_attention_heads=8, use_cpu_initialization=True)
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs)
    model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=128,
        max_sequence_length=16,
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
    SwapDistributedOptimizer.swap_optimizer_times = 16
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
    SwapOptimizerMixin._swap_optimizer_times = 0


class Timers:
    def __init__(self, *args, **kwargs):
        self._dummy_timer = DummyTimer()

    def __call__(self, *args, **kwargs):
        return self._dummy_timer


def run_optimizer_steps(optimizer, steps=FAST_OPTIMIZER_STEPS, overlap_param_gather=False, muon=False):
    """Run a small deterministic optimizer workload shared by the swap tests."""
    for _ in range(steps):
        if muon:
            for sub_optimizer in optimizer.chained_optimizers:
                for float16_group in sub_optimizer.float16_groups:
                    for param in float16_group:
                        param.grad = torch.randn_like(param.data, dtype=param.data.dtype)
        else:
            for float16_group in optimizer.chained_optimizers[0].model_float16_groups:
                for param in float16_group:
                    param.grad = torch.randn_like(param.data, dtype=param.data.dtype)
        optimizer.step()
        if overlap_param_gather:
            for model_chunk in optimizer.model_chunks:
                model_chunk.start_param_sync(force_sync=True)
        torch.cuda.synchronize()


def clone_optimizer_model_params(optimizer, muon=False):
    group_attr = "float16_groups" if muon else "model_float16_groups"
    groups = getattr(optimizer.chained_optimizers[0], group_attr)
    return [param.detach().clone() for param in itertools.chain(*groups)]


class SwapOptimizerTestBase(DistributedTest):
    topologies = ()

    def test_swap_optimizer(self):
        for tensor_parallel_size, pipeline_parallel_size, overlap_grad_reduce, overlap_param_gather in self.topologies:
            args = parse_args(None, True)
            args.npu_deterministic = False
            args.overlap_grad_reduce = overlap_grad_reduce
            args.overlap_param_gather = overlap_param_gather
            set_args(args)

            reset_swap_distributed_optimizer()
            init_mock_args(args, use_distributed_optimizer=True)
            initialize_model_parallel(
                tensor_model_parallel_size=tensor_parallel_size,
                pipeline_model_parallel_size=pipeline_parallel_size,
            )
            _, optimizer = setup_model_and_optimizer(seed=5, use_distributed_optimizer=True)
            run_optimizer_steps(optimizer, overlap_param_gather=overlap_param_gather)
            truth_params = clone_optimizer_model_params(optimizer)

            reset_swap_distributed_optimizer()
            init_mock_args(args, use_distributed_optimizer=True, swap_optimizer=True)
            initialize_model_parallel(
                tensor_model_parallel_size=tensor_parallel_size,
                pipeline_model_parallel_size=pipeline_parallel_size,
            )
            _, optimizer = setup_model_and_optimizer(seed=5, use_distributed_optimizer=True)
            run_optimizer_steps(optimizer, overlap_param_gather=overlap_param_gather)
            swap_optimizer_params = clone_optimizer_model_params(optimizer)

            for param, swap_optimizer_param in zip(truth_params, swap_optimizer_params):
                assert torch.allclose(param, swap_optimizer_param, rtol=0.005, atol=0.005)


class TestDistributedOptimizer(SwapOptimizerTestBase):
    world_size = 2
    topologies = ((1, 1, False, False),)

    def test_swap_optimizer_deferred_release(self):
        """Verify swap_optimizer_times=0 (deferred release) produces the same
        results as swap_optimizer_times=16 (default mode).
        """
        from mindspeed.core.optimizer.swap_optimizer.swap_optimizer import SwapDistributedOptimizer

        tp_pp = (1, 1)
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
        run_optimizer_steps(optimizer)
        baseline_params = clone_optimizer_model_params(optimizer)

        # Deferred release: swap_optimizer with times=0
        reset_swap_distributed_optimizer()
        init_mock_args(args, use_distributed_optimizer=True, swap_optimizer=True, swap_optimizer_times=0)
        repatch(vars(args))
        initialize_model_parallel(tensor_model_parallel_size=tp_pp[0], pipeline_model_parallel_size=tp_pp[1])
        _, optimizer = setup_model_and_optimizer(seed=5, use_distributed_optimizer=True)
        run_optimizer_steps(optimizer)
        deferred_params = clone_optimizer_model_params(optimizer)

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

        tp_pp = (1, 1)
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
        run_optimizer_steps(optimizer, muon=True)
        baseline_params = clone_optimizer_model_params(optimizer, muon=True)

        # Deferred release: swap_optimizer with times=0
        init_mock_args(
            args, use_distributed_optimizer=True, swap_optimizer=True, swap_optimizer_times=0, optimizer='muon'
        )
        reset_swap_optimizer_mixin()
        initialize_model_parallel(tensor_model_parallel_size=tp_pp[0], pipeline_model_parallel_size=tp_pp[1])
        _, optimizer = setup_model_and_muon_optimizer(seed=5)
        run_optimizer_steps(optimizer, muon=True)
        deferred_params = clone_optimizer_model_params(optimizer, muon=True)

        # Verify numerical consistency
        for p, dp in zip(baseline_params, deferred_params):
            assert torch.allclose(p.data, dp.data, rtol=0.005, atol=0.005)

        # Verify class state is properly reset after each iteration
        assert SwapOptimizerMixin._step_count == 0

    def test_swap_muon_checkpoint_round_trip(self, class_tmpdir):
        """Swap-Muon checkpoints restore parameters and momentum without retaining NPU storage."""
        from mindspeed.core.optimizer.swap_muon.swap_muon import SwapOptimizerMixin

        args = parse_args(None, True)
        args.npu_deterministic = False
        args.overlap_grad_reduce = False
        args.overlap_param_gather = False
        set_args(args)
        init_mock_args(
            args,
            use_distributed_optimizer=True,
            swap_optimizer=True,
            swap_optimizer_times=0,
            optimizer='muon',
        )
        repatch(vars(args))

        reset_swap_optimizer_mixin()
        try:
            initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
            _, optimizer = setup_model_and_muon_optimizer(seed=5)
            run_optimizer_steps(optimizer, steps=1, muon=True)

            checkpoint_path = os.path.join(str(class_tmpdir), f"swap_muon_optimizer_rank_{dist.get_rank()}.pt")
            optimizer.save_state_dict_to_file(checkpoint_path)
            expected_cpu_states = {
                param: {
                    key: value.clone() if value is not None else None
                    for key, value in cpu_states.items()
                    if key in ("param", "momentum_buffer")
                }
                for param, cpu_states in SwapOptimizerMixin._param_to_cpu_states.items()
            }
            assert expected_cpu_states
            assert all(param.storage().size() == 0 for param in expected_cpu_states)

            for cpu_states in SwapOptimizerMixin._param_to_cpu_states.values():
                cpu_states["param"].add_(1)
                cpu_states["momentum_buffer"].add_(1)

            optimizer.load_state_dict_from_file(checkpoint_path)

            for param, expected in expected_cpu_states.items():
                cpu_states = SwapOptimizerMixin._param_to_cpu_states[param]
                assert param.storage().size() == 0
                assert SwapOptimizerMixin._state_map[param]["momentum_buffer"].storage().size() == 0
                assert torch.equal(cpu_states["param"], expected["param"])
                assert torch.equal(cpu_states["momentum_buffer"], expected["momentum_buffer"])
        finally:
            reset_swap_optimizer_mixin()


@pytest.mark.slow
class TestDistributedOptimizerTopologies(SwapOptimizerTestBase):
    world_size = 8
    topologies = tuple(
        (tensor_parallel_size, pipeline_parallel_size, overlap_grad_reduce, overlap_param_gather)
        for tensor_parallel_size, pipeline_parallel_size in ((4, 1), (2, 2), (8, 1))
        for overlap_grad_reduce in (True, False)
        for overlap_param_gather in (True, False)
    )
