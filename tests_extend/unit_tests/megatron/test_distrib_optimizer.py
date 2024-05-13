import copy
import torch
import torch_npu  # noqa
import ascendspeed.megatron_adaptor  # noqa
from apex.optimizers import FusedAdam as Adam

from megatron.core import DistributedDataParallel as DDP
from megatron.core.transformer import TransformerConfig, MegatronModule
from megatron.core.parallel_state import get_data_parallel_group
from megatron.global_vars import set_args
from megatron.timers import DummyTimer
from megatron.optimizer import (
    DistributedOptimizer,
    Float16OptimizerWithFloat16Params,
    ConstantGradScaler,
)

from commons import set_random_seed, initialize_model_parallel
from unit_tests.common import DistributedTest


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


class DummyArgs:
    no_gradient_accumulation_fusion = True
    use_distributed_optimizer = True
    overlap_param_gather = False
    barrier_with_L1_time = False
    fp16 = True
    reuse_fp32_param = False


def step_optimizer(model, use_distributed: bool, seed: int = None):
    args = DummyArgs()
    set_args(args)
    set_random_seed(seed)

    model = torch.nn.ModuleList(
        [
            DDP(
                model_chunk.config,
                model_chunk,
                data_parallel_group=get_data_parallel_group(with_context_parallel=True),
                accumulate_allreduce_grads_in_fp32=True,
                overlap_grad_reduce=False,
                use_distributed_optimizer=use_distributed,
            )
            for model_chunk in model
        ]
    )

    # Params initialization
    for p in model.parameters():
        p.data = torch.arange(p.numel(), dtype=torch.float16).reshape(p.data.shape)

    model = model.cuda()

    opt_ty = (
        DistributedOptimizer if use_distributed else Float16OptimizerWithFloat16Params
    )
    optim = opt_ty(
        Adam(model.parameters()),
        clip_grad=1,
        log_num_zeros_in_grad=False,
        check_for_nan_in_grad=False,
        params_have_main_grad=True,
        fp16=args.fp16,
        bf16=False,
        params_dtype=torch.float16,
        grad_scaler=ConstantGradScaler(1.0),
        models=model,
    )

    for _ in range(500):
        # Force optimizer state initialization
        for p in model.parameters():
            p.grad = torch.randn_like(p.data, dtype=p.data.dtype)
        # Update params
        optim.step(args, Timers())

    return copy.deepcopy(list(model.parameters()))


class TestDistributedOptimizer(DistributedTest):
    world_size = 8

    def test_distributed_optimizer(self):
        initialize_model_parallel(1, 1)

        config = TransformerConfig(
            num_layers=2,
            hidden_size=8,
            num_attention_heads=4,
            use_cpu_initialization=True,
            fp16=True,
        )
        model = [Model(config)]

        params = step_optimizer(model, use_distributed=False, seed=123)
        dist_params = step_optimizer(model, use_distributed=True, seed=123)

        for p, dist_p in zip(params, dist_params):
            assert torch.allclose(p.data, dist_p.data)
