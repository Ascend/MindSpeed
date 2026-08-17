# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from functools import partial
import copy
import itertools

import pytest
import torch

from mindspeed import megatron_adaptor  # noqa: F401
from megatron.training.arguments import parse_args
from megatron.training.global_vars import get_args, set_args
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.timers import DummyTimer
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.training.training import get_model
from megatron.core.utils import unwrap_model

from mindspeed.core.optimizer.swap_optimizer.swap_optimizer import (
    SwapDistributedOptimizer,
)
from tests_extend.unit_tests.common import DistributedTest
from tests_extend.commons import set_random_seed, initialize_model_parallel


def initialize_gpt_model(
    pre_process=True,
    post_process=True,
    seed=0,
    config=None,
    pg_collection=None,
    vp_stage=None,
    **config_kwargs,
):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    if config is None:
        default_config_kwargs = dict(
            num_layers=8,
            hidden_size=512,
            num_attention_heads=32,
            use_cpu_initialization=True,
        )
        default_config_kwargs.update(**config_kwargs)
        config = TransformerConfig(**default_config_kwargs)
    config.gradient_accumulation_fusion = False
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=1024,
        max_sequence_length=64,
        pre_process=pre_process,
        post_process=post_process,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )

    model.bfloat16()
    with torch.no_grad():
        for p in model.parameters():
            p.random_()
    return model


def init_mock_args(args, use_distributed_optimizer=False, swap_optimizer=False):
    args.data_parallel_random_init = False
    args.virtual_pipeline_model_parallel_size = None
    args.fp16 = False
    args.bf16 = True
    args.accumulate_allreduce_grads_in_fp32 = True
    args.use_distributed_optimizer = use_distributed_optimizer
    args.use_layer_wise_distributed_optimizer = False
    args.optimizer = "adam"
    args.ddp_bucket_size = None
    args.swap_optimizer = swap_optimizer
    args.num_query_groups = None
    return args


def setup_model_and_optimizer(seed, use_distributed_optimizer=False):
    model = get_model(partial(initialize_gpt_model, seed=seed, bf16=True))
    set_random_seed(seed)
    args = get_args()
    config = OptimizerConfig(
        lr=1e-4,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=use_distributed_optimizer,
        overlap_param_gather=args.overlap_param_gather,
        barrier_with_L1_time=False,
    )
    config.timers = Timers()
    optimizer = get_megatron_optimizer(config, model)
    optimizer.reload_model_params()
    return unwrap_model(model), optimizer


def set_random_grads(optimizer):
    for model_group in optimizer.chained_optimizers[0].model_float16_groups:
        for param in model_group:
            if hasattr(param, "main_grad"):
                param.main_grad.copy_(torch.randn_like(param.main_grad))
            else:
                param.grad = torch.randn_like(param)


def reset_swap_optimizer_state():
    SwapDistributedOptimizer.ALL_OPTIMIZER.clear()
    SwapDistributedOptimizer.param_to_cpu_states_map.clear()
    SwapDistributedOptimizer.param_to_device_states_map.clear()
    SwapDistributedOptimizer.main_param_to_model_param_map.clear()
    SwapDistributedOptimizer.no_swap_params.clear()
    SwapDistributedOptimizer.swap_to_device_events_map.clear()
    SwapDistributedOptimizer.swap_to_host_events_map.clear()
    SwapDistributedOptimizer.copy_to_model_param_events_map.clear()
    SwapDistributedOptimizer.swap_to_device_stream = None
    SwapDistributedOptimizer.swap_to_host_stream = None


class Timers:
    def __init__(self, *args, **kwargs):
        self._dummy_timer = DummyTimer()

    def __call__(self, *args, **kwargs):
        return self._dummy_timer


class TestDistributedOptimizer(DistributedTest):
    world_size = 8

    @pytest.mark.parametrize("is_deterministic", [False])
    @pytest.mark.parametrize(
        "overlap_grad_reduce", [pytest.param(True, marks=pytest.mark.slow), False]
    )
    @pytest.mark.parametrize(
        "overlap_param_gather", [pytest.param(True, marks=pytest.mark.slow), False]
    )
    @pytest.mark.parametrize(
        "tp_pp",
        [
            pytest.param((4, 1), marks=pytest.mark.slow),
            (2, 2),
            pytest.param((8, 1), marks=pytest.mark.slow),
        ],
    )
    def test_swap_optimizer(
        self, tp_pp, is_deterministic, overlap_grad_reduce, overlap_param_gather
    ):
        from mindspeed.megatron_adaptor import repatch

        args = parse_args(None, True)
        args.npu_deterministic = is_deterministic
        args.overlap_grad_reduce = overlap_grad_reduce
        args.overlap_param_gather = overlap_param_gather
        init_mock_args(args, use_distributed_optimizer=True)
        set_args(args)

        # truth
        repatch(
            {
                "optimizer": "adam",
                "swap_optimizer": False,
                "use_distributed_optimizer": True,
                "use_layer_wise_distributed_optimizer": False,
            }
        )
        initialize_model_parallel(
            tensor_model_parallel_size=tp_pp[0], pipeline_model_parallel_size=tp_pp[1]
        )
        _, optimizer = setup_model_and_optimizer(seed=5, use_distributed_optimizer=True)
        for _ in range(10):
            set_random_grads(optimizer)
            optimizer.step()
            if overlap_param_gather:
                for model_chunk in optimizer.model_chunks:
                    model_chunk.start_param_sync(force_sync=True)
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        truth_params = copy.deepcopy(
            list(itertools.chain(*optimizer.chained_optimizers[0].model_float16_groups))
        )

        # swap_optimizer
        init_mock_args(args, use_distributed_optimizer=True, swap_optimizer=True)
        reset_swap_optimizer_state()
        repatch(
            {
                "optimizer": "adam",
                "swap_optimizer": True,
                "use_distributed_optimizer": True,
                "use_layer_wise_distributed_optimizer": False,
            }
        )
        initialize_model_parallel(
            tensor_model_parallel_size=tp_pp[0], pipeline_model_parallel_size=tp_pp[1]
        )
        _, optimizer = setup_model_and_optimizer(seed=5, use_distributed_optimizer=True)
        for _ in range(10):
            set_random_grads(optimizer)
            optimizer.step()
            if overlap_param_gather:
                for model_chunk in optimizer.model_chunks:
                    model_chunk.start_param_sync(force_sync=True)
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        swap_optimizer_params = copy.deepcopy(
            list(itertools.chain(*optimizer.chained_optimizers[0].model_float16_groups))
        )

        for p, swap_optimizer_p in zip(truth_params, swap_optimizer_params):
            if is_deterministic:
                assert torch.allclose(p.data, swap_optimizer_p.data, rtol=0, atol=0)
            else:
                assert torch.allclose(
                    p.data, swap_optimizer_p.data, rtol=0.005, atol=0.005
                )

        reset_swap_optimizer_state()
        repatch({"swap_optimizer": False})
