import copy
import pytest
import torch
import torch_npu  # noqa: F401

import mindspeed.megatron_adaptor  # noqa: F401

from megatron.core import DistributedDataParallel as DDP
from megatron.core.transformer import TransformerConfig, MegatronModule
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.core.timers import DummyTimer
from megatron.core.optimizer import (
    OptimizerConfig,
    Float16OptimizerWithFloat16Params,
    ConstantGradScaler,
)

from tests_extend.commons import set_random_seed, initialize_model_parallel
from tests_extend.unit_tests.common import DistributedTest

from mindspeed.core.optimizer.low_precision.quant_adamw import AdamW as QuantAdamW


class Model(MegatronModule):
    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(in_features=8, out_features=2)

    def forward(self, x):
        return self.linear(x)


class Timers:
    def __init__(self, *args, **kwargs):
        self._dummy_timer = DummyTimer()

    def __call__(self, *args, **kwargs):
        return self._dummy_timer


def _build_optimizer(model: torch.nn.Module, optimizer_config: OptimizerConfig):
    base_optim = QuantAdamW(model.parameters())
    grad_scaler = ConstantGradScaler(1.0)

    def init_state_fn(opt):
        for group in opt.param_groups:
            for p in group['params']:
                st = opt.state.get(p, None)
                if st is None or len(st) == 0:
                    opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                    opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)

    return Float16OptimizerWithFloat16Params(
        base_optim,
        optimizer_config,
        grad_scaler,
        init_state_fn,
    )


def _step_optimizer(model_chunks, ddp_config, optimizer_config, seed: int = None):
    set_random_seed(seed)

    # Wrap with Megatron DDP
    ddp_wrapped = torch.nn.ModuleList(
        [
            DDP(
                model_chunk.config,
                ddp_config,
                model_chunk,
            )
            for model_chunk in model_chunks
        ]
    )

    # Move to device and cast params to desired precision
    ddp_wrapped = ddp_wrapped.cuda()
    # Infer precision choice from optimizer_config
    if optimizer_config.fp16:
        ddp_wrapped = ddp_wrapped.half()
    elif optimizer_config.bf16:
        ddp_wrapped = ddp_wrapped.bfloat16()

    # Initialize params deterministically with correct dtype
    for p in ddp_wrapped.parameters():
        p.data = torch.arange(p.numel(), dtype=p.dtype, device=p.device).reshape(p.data.shape)

    optim = _build_optimizer(ddp_wrapped, optimizer_config)

    for _ in range(10):
        # Force optimizer state initialization and run step
        for p in ddp_wrapped.parameters():
            p.grad = torch.randn_like(p.data, dtype=p.data.dtype)
        optim.step()

    return copy.deepcopy(list(ddp_wrapped.parameters()))


class TestQuantGrads(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("fp16_bf16", [(True, False), (False, True)])
    def test_quantized_grads_match_baseline(self, fp16_bf16):
        (fp16, bf16) = fp16_bf16

        args = parse_args(None, True)
        args.fp16 = fp16
        args.bf16 = bf16
        # Default: no quant flags
        args.quant_grads = False
        set_args(args)

        initialize_model_parallel(1, 1)

        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=4,
            use_cpu_initialization=True,
            fp16=fp16,
            bf16=bf16,
        )
        model = [Model(config)]

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=False,
            use_distributed_optimizer=False,
            check_for_nan_in_grad=False,
        )

        optimizer_config = OptimizerConfig(
            clip_grad=1.0,
            fp16=fp16,
            bf16=bf16,
            barrier_with_L1_time=False,
            overlap_param_gather=False,
        )
        timers = Timers()
        optimizer_config.timers = timers

        # Baseline path without quantized grads
        baseline_params = _step_optimizer(model, ddp_config, optimizer_config, seed=123)

        # Enable quant grads via repatching features
        from mindspeed.megatron_adaptor import repatch

        repatch({'quant_grads': True})
        quant_params = _step_optimizer(model, ddp_config, optimizer_config, seed=123)

        for p, q in zip(baseline_params, quant_params):
            assert torch.allclose(p.data.float().cpu(), q.data.float().cpu(), atol=0.01, rtol=0.01)
