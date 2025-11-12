import copy
import pytest
import torch
import torch_npu  # noqa: F401

import mindspeed.megatron_adaptor  # noqa: F401

from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from megatron.core.timers import DummyTimer
from megatron.core.optimizer import (
    OptimizerConfig,
    Float16OptimizerWithFloat16Params,
    ConstantGradScaler,
)

from tests_extend.commons import set_random_seed, initialize_model_parallel
from tests_extend.unit_tests.common import DistributedTest

from mindspeed.core.optimizer.low_precision.quant_adamw import AdamW as QuantAdamW


class Timers:
    def __init__(self, *args, **kwargs):
        self._dummy_timer = DummyTimer()

    def __call__(self, *args, **kwargs):
        return self._dummy_timer


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x):
        return self.linear(x)


def _build_optimizer(model: torch.nn.Module, optimizer_config: OptimizerConfig):
    # Base optimizer is MindSpeed quant-aware AdamW, which reads quant flags from global args.
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


def _step_optimizer(model: torch.nn.Module, optimizer_config: OptimizerConfig, seed: int = None):
    set_random_seed(seed)

    # Deterministic parameter initialization.
    for p in model.parameters():
        p.data = torch.arange(p.numel(), dtype=p.dtype, device=p.device).reshape(p.data.shape)

    optim = _build_optimizer(model, optimizer_config)

    for _ in range(10):
        # Initialize grads deterministically for reproducibility
        for p in model.parameters():
            p.grad = torch.randn_like(p.data, dtype=p.data.dtype)
        # Update
        optim.step()

    return copy.deepcopy(list(model.parameters()))


class TestQuantStates(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("fp16_bf16,quant_token", [
        ((True, False), "fp8"),
        ((True, False), "hif8"),
        ((True, False), "mxfp8"),
        ((False, True), "fp8"),
        ((False, True), "hif8"),
        ((False, True), "mxfp8"),
        ((True, False), "fp16"),
        ((False, True), "fp16"),
    ])
    def test_quant_states_optimizer_matches_baseline(self, fp16_bf16, quant_token):
        (fp16, bf16) = fp16_bf16

        # Prepare args for optimizer to read quant flags
        args = parse_args(None, True)
        args.fp16 = fp16
        args.bf16 = bf16
        set_args(args)

        initialize_model_parallel(1, 1)

        model = Model().cuda()
        model = model.half() if fp16 else model.bfloat16()

        optimizer_config = OptimizerConfig(
            clip_grad=1.0,
            fp16=fp16,
            bf16=bf16,
        )
        timers = Timers()
        optimizer_config.timers = timers

        # Baseline without quantized states
        setattr(args, 'quant_states', None)
        set_args(args)
        baseline_params = _step_optimizer(model, optimizer_config, seed=123)

        # Quantized states path
        setattr(args, 'quant_states', quant_token)
        set_args(args)
        quant_params = _step_optimizer(model, optimizer_config, seed=123)

        for p, q in zip(baseline_params, quant_params):
            assert torch.allclose(p.data.float().cpu(), q.data.float().cpu(), atol=0.01, rtol=0.01)

