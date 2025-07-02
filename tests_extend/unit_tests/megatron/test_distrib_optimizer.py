import dataclasses
import copy
import pytest
import torch
import torch_npu
import mindspeed.megatron_adaptor
from apex.optimizers import FusedAdam as Adam
from commons import set_random_seed, initialize_model_parallel
from unit_tests.common import DistributedTest
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.transformer import TransformerConfig, MegatronModule
from megatron.core.parallel_state import get_data_parallel_group
from megatron.training.global_vars import set_args, get_args, get_timers, _set_timers
from megatron.training.arguments import parse_args
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.core.utils import get_model_config


class Model(MegatronModule):
    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(in_features=8, out_features=2)

    def forward(self, x):
        return self.linear(x)


def step_optimizer(model, use_distributed: bool, seed: int = None, 
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0):
    set_random_seed(seed)
    args = get_args()
    config = get_model_config(model[0])
    ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32,
            overlap_grad_reduce=args.overlap_grad_reduce,
            use_distributed_optimizer=args.use_distributed_optimizer,
            check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad,
            bucket_size=args.ddp_bucket_size,
            average_in_collective=args.ddp_average_in_collective)
    model = torch.nn.ModuleList([DDP(config,
                 ddp_config,
                 model_chunk,
                 # Turn off bucketing for model_chunk 2 onwards, since communication for these
                 # model chunks is overlapped with compute anyway.
                 disable_bucketing=(model_chunk_idx > 0))
             for (model_chunk_idx, model_chunk) in enumerate(model)])

    # Params initialization
    for p in model.parameters():
        p.data = torch.arange(p.numel(), dtype=torch.float16).reshape(p.data.shape)

    model = model.cuda()

    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = get_timers()
    optimizer = get_megatron_optimizer(config, model, no_wd_decay_cond,
                                       scale_lr_cond, lr_mult)

    for _ in range(500):
        # Force optimizer state initialization
        for p in model.parameters():
            p.grad = torch.randn_like(p.data, dtype=p.data.dtype)
        # Update params
        optimizer.step()

    return copy.deepcopy(list(model.parameters()))


class TestDistributedOptimizer(DistributedTest):
    world_size = 8
    args = parse_args(None, True)
    args.no_gradient_accumulation_fusion = True
    args.use_distributed_optimizer = True
    args.overlap_param_gather = False
    args.barrier_with_L1_time = False
    args.fp16 = True
    args.reuse_fp32_param = False
    args.lr = 1e-6
    set_args(args)
    _set_timers(args)

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
