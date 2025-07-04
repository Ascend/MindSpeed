import copy
import pytest
from unit_tests.common import DistributedTest
from commons import set_random_seed, initialize_model_parallel
import torch
import torch_npu  # noqa

from apex.optimizers import FusedAdam as Adam
from megatron.legacy.model import Float16Module
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from megatron.core.timers import DummyTimer
from megatron.core.optimizer import (
    OptimizerConfig,
    Float16OptimizerWithFloat16Params,
    ConstantGradScaler,
)

import mindspeed.megatron_adaptor  # noqa


class Timers:
    def __init__(self, *args, **kwargs):
        self._dummy_timer = DummyTimer()

    def __call__(self, *args, **kwargs):
        return self._dummy_timer


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(in_features=256, out_features=256)

    def forward(self, x):
        return self.linear(x)


def step_optimizer(model, optimizer_config, seed: int = None):
    set_random_seed(seed)

    # Params initialization
    for p in model.parameters():
        p.data = torch.arange(p.numel(), dtype=p.dtype, device=p.device).reshape(p.data.shape)

    def init_state_fn(opt):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                    opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)

    optimizer = Adam(model.parameters())
    grad_scaler = ConstantGradScaler(1.0)
    optim = Float16OptimizerWithFloat16Params(
        optimizer,
        optimizer_config,
        grad_scaler,
        init_state_fn,
    )

    for _ in range(10):
        # Force optimizer state initialization
        for p in model.parameters():
            p.grad = torch.randn_like(p.data, dtype=p.data.dtype)
        # Update params
        optim.step()

    return copy.deepcopy(list(model.parameters()))


class TestQuantOptimizer(DistributedTest):
    world_size = 1
    
    @pytest.mark.skip(reason='not support for current version')
    @pytest.mark.parametrize("fp16_bf16_reuse_param", [(True, False, False), (False, True, False), (False, True, True)])
    def test_fp8_optimizer(self, fp16_bf16_reuse_param):
        (fp16, bf16, reuse_param) = fp16_bf16_reuse_param
        args = parse_args(None, True)
        args.fp16 = fp16
        args.bf16 = bf16
        args.reuse_fp32_param = reuse_param
        set_args(args)

        initialize_model_parallel(1, 1)

        model = Model()
        model = model.cuda()
        model = Float16Module(model, args)

        optimizer_config = OptimizerConfig(
            clip_grad=1,
            fp16=fp16,
            bf16=bf16)
        timers = Timers()
        optimizer_config.timers = timers

        params = step_optimizer(model, optimizer_config, seed=123)
        args.quant_states = "fp8"
        set_args(args)
        quant_params = step_optimizer(model, optimizer_config, seed=123)

        for p, dist_p in zip(params, quant_params):
            assert torch.allclose(p.data.float().cpu(), dist_p.data.float().cpu(), atol=0.005, rtol=0.005)
    
    @pytest.mark.skip(reason='not support for current version')
    @pytest.mark.parametrize("fp16_bf16_reuse_param", [(True, False, False), (False, True, False), (False, True, True)])
    def test_hif8_optimizer(self, fp16_bf16_reuse_param):
        (fp16, bf16, reuse_param) = fp16_bf16_reuse_param
        args = parse_args(None, True)
        args.fp16 = fp16
        args.bf16 = bf16
        args.reuse_fp32_param = reuse_param
        set_args(args)

        initialize_model_parallel(1, 1)

        model = Model()
        model = model.cuda()
        model = Float16Module(model, args)

        optimizer_config = OptimizerConfig(
            clip_grad=1,
            fp16=fp16,
            bf16=bf16)
        timers = Timers()
        optimizer_config.timers = timers

        params = step_optimizer(model, optimizer_config, seed=123)
        args.quant_states = "hif8"
        set_args(args)
        quant_params = step_optimizer(model, optimizer_config, seed=123)

        for p, dist_p in zip(params, quant_params):
            assert torch.allclose(p.data.float().cpu(), dist_p.data.float().cpu(), atol=0.005, rtol=0.005)
    
    @pytest.mark.skip(reason='not support for current version')
    @pytest.mark.parametrize("fp16_bf16_reuse_param", [(True, False, False), (False, True, False), (False, True, True)])
    def test_mxfp8_optimizer(self, fp16_bf16_reuse_param):
        (fp16, bf16, reuse_param) = fp16_bf16_reuse_param
        args = parse_args(None, True)
        args.fp16 = fp16
        args.bf16 = bf16
        args.reuse_fp32_param = reuse_param
        set_args(args)

        initialize_model_parallel(1, 1)

        model = Model()
        model = model.cuda()
        model = Float16Module(model, args)

        optimizer_config = OptimizerConfig(
            clip_grad=1,
            fp16=fp16,
            bf16=bf16)
        timers = Timers()
        optimizer_config.timers = timers

        params = step_optimizer(model, optimizer_config, seed=123)
        args.quant_states = "mxfp8"
        set_args(args)
        quant_params = step_optimizer(model, optimizer_config, seed=123)

        for p, dist_p in zip(params, quant_params):
            assert torch.allclose(p.data.float().cpu(), dist_p.data.float().cpu(), atol=0.005, rtol=0.005)
