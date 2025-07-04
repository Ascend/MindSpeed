import copy
import pytest
from unit_tests.common import DistributedTest
from commons import set_random_seed, initialize_model_parallel
import torch
import torch_npu  # noqa
from apex.optimizers import FusedAdam as Adam

from megatron.core import DistributedDataParallel as DDP
from megatron.core.transformer import TransformerConfig, MegatronModule
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.core.timers import DummyTimer
from megatron.core.optimizer import (
    DistributedOptimizer,
    Float16OptimizerWithFloat16Params,
    ConstantGradScaler,
    OptimizerConfig
)

import mindspeed.megatron_adaptor  # noqa


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


def step_optimizer(model, ddp_config, optimizer_config, seed: int = None):
    set_random_seed(seed)

    model = torch.nn.ModuleList(
        [
            DDP(
                model_chunk.config,
                ddp_config,
                model_chunk,
            )
            for model_chunk in model
        ]
    )

    # Params initialization
    for p in model.parameters():
        p.data = torch.arange(p.numel(), dtype=torch.float16).reshape(p.data.shape)

    model = model.cuda()

    def init_state_fn(opt):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                    opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)

    grad_scaler = ConstantGradScaler(1.0)
    optim = Float16OptimizerWithFloat16Params(
        Adam(model.parameters()),
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


class TestQuantGrads(DistributedTest):
    world_size = 4

    @pytest.mark.skip(reason='not support for current version')
    @pytest.mark.parametrize("fp16_bf16", [(True, False), (False, True)])
    def test_distributed_optimizer(self, fp16_bf16):
        (fp16, bf16) = fp16_bf16
        args = parse_args(None, True)
        args.fp16 = fp16
        args.bf16 = bf16
        args.no_gradient_accumulation_fusion = True
        set_args(args)

        initialize_model_parallel(1, 1)

        config = TransformerConfig(
            num_layers=2,
            hidden_size=8,
            num_attention_heads=4,
            use_cpu_initialization=True,
            fp16=fp16,
            bf16=bf16
        )
        model = [Model(config)]

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32,
            overlap_grad_reduce=args.overlap_grad_reduce,
            use_distributed_optimizer=args.use_distributed_optimizer,
            check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad,
            bucket_size=args.ddp_bucket_size
        )

        optimizer_config = OptimizerConfig(
            clip_grad=1,
            fp16=fp16,
            bf16=bf16,
            barrier_with_L1_time=False,
            overlap_param_gather=False,
        )
        timers = Timers()
        optimizer_config.timers = timers

        args.quant_grads = False
        set_args(args)
        params = step_optimizer(model, ddp_config, optimizer_config, seed=123)
        args.quant_grads = True
        set_args(args)
        quant_params = step_optimizer(model, ddp_config, optimizer_config, seed=123)

        for p, dist_p in zip(params, quant_params):
            assert torch.allclose(p.data.float().cpu(), dist_p.data.float().cpu(), atol=0.005, rtol=0.005)
