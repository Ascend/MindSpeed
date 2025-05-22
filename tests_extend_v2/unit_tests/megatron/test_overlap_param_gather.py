import copy
import pytest
import torch
import mindspeed.megatron_adaptor_v2
from apex.optimizers import FusedAdam as Adam

from types import SimpleNamespace
from megatron.core import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.transformer import TransformerConfig, MegatronModule
from megatron.training.global_vars import set_args, get_timers, set_global_variables
from megatron.training.arguments import parse_args
from megatron.core.timers import DummyTimer
from megatron.core.optimizer import (
    DistributedOptimizer,
    ConstantGradScaler,
    OptimizerConfig,
)
from megatron.core import mpu
from tests_extend.commons import set_random_seed, initialize_model_parallel
from tests_extend.unit_tests.common import DistributedTest


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


def create_test_args():
    args = parse_args(None, True)
    args.no_gradient_accumulation_fusion = True
    args.use_distributed_optimizer = True
    args.overlap_grad_reduce = True
    args.barrier_with_L1_time = False
    args.reuse_fp32_param = False
    args.accumulation_allreudce_grads_in_fp32 = True
    return args


def step_optimizer(model, optimizer_config, ddp_config, seed: int = None):
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

    opt_ty = DistributedOptimizer

    def init_state_fn(opt):
        for group in opt.param_groups:
            for p in group['param']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                    opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)

    grad_scaler = ConstantGradScaler(1.0)
    optimizer_args = [
        Adam(model.parameters()),
        optimizer_config,
        grad_scaler,
        init_state_fn,
        model,
    ]
    model_parallel_rank = torch.distributed.get_rank(mpu.get_model_parallel_group())
    per_model_buffers = {}


    for model_idx, model_chunk in enumerate(model):
        if hasattr(model_chunk, 'buffers'):
            per_model_buffers[model_idx] = model_chunk.buffers

    optim = opt_ty(
        *optimizer_args,
        per_model_buffers=per_model_buffers,
        data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
        data_parallel_group_gloo=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
        data_parallel_group_idx=model_parallel_rank,
        distributed_optimizer_instance_id=0
    )

    for _ in range(500):
        # Force optimizer state initialization
        for p in model.parameters():
            p.grad = torch.randn_like(p.data, dtype=p.data.dtype)
        # Update params
        optim.step()

    return copy.deepcopy(list(model.parameters()))


class TestOverlapParamGather(DistributedTest):
    world_size = 8

    @pytest.mark.parametrize("fp16_bf16", [(True, False), (False, True)])
    def test_overlap_param_gather(self, fp16_bf16):
        (fp16, bf16) = fp16_bf16
        args = create_test_args()
        if bf16:
            args.fp16 = fp16
            args.bf16 = bf16
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
            grad_reduce_in_fp32=args.accumulation_allreudce_grads_in_fp32,
            overlap_grad_reduce=args.overlap_grad_reduce,
            use_distributed_optimizer=args.use_distributed_optimizer,
            check_for_nan_in_grad=False,
        )

        optimizer_config = OptimizerConfig(
            clip_grad=1,
            fp16=fp16,
            bf16=bf16,
            barrier_with_L1_time=False,
            overlap_param_gather_with_optimizer_step=False,
        )
        timers = Timers()
        optimizer_config.timers = timers

        params = step_optimizer(model, optimizer_config, ddp_config, seed=123)

        optimizer_config.overlap_param_gather_with_optimizer_step = True

        dist_params = step_optimizer(model, optimizer_config, ddp_config, seed=123)

        for p, dist_p in zip(params, dist_params):
            assert torch.allclose(p.data, dist_p.data)
