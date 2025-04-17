# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from functools import wraps

import torch
import torch_npu

from megatron.training import get_args
from mindspeed.functional.tflops_calculate.tflops_utils import set_count, get_flops_counter
from mindspeed.core.auto_parallel.auto_parallel_profiling import Profiling, OperateProfile


def train_wrapper(train):
    @wraps(train)
    def wrapper(*args, **kwargs):
        args_ = get_args()
        if args_.profile:
            args_.profile_npu = True
            args_.profile = False
        else:
            args_.profile_npu = False

        is_profile = hasattr(args_, 'profile_npu') and args_.profile_npu \
                and ((torch.distributed.get_rank() in args_.profile_ranks) or (-1 in args_.profile_ranks))
        if is_profile:
            active = args_.profile_step_end - args_.profile_step_start
            skip_first = args_.profile_step_start

            if args_.profile_with_cpu:
                activities = [torch_npu.profiler.ProfilerActivity.NPU, torch_npu.profiler.ProfilerActivity.CPU]
            else:
                activities = [torch_npu.profiler.ProfilerActivity.NPU]

            level2level = {
                'level0': torch_npu.profiler.ProfilerLevel.Level0,
                'level1': torch_npu.profiler.ProfilerLevel.Level1,
                'level2': torch_npu.profiler.ProfilerLevel.Level2
            }
            profiler_level = level2level[args_.profile_level]

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=profiler_level,
                l2_cache=False
            )

            with torch_npu.profiler.profile(
                activities=activities,
                record_shapes=args_.profile_record_shapes,
                profile_memory=args_.profile_with_memory,
                with_stack=args_.profile_with_stack,
                experimental_config=experimental_config,
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=active, repeat=1, skip_first=skip_first),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args_.profile_save_path)
            ) as prof:
                args_.prof = prof
                return train(*args, **kwargs)
        return train(*args, **kwargs)

    return wrapper


def train_step_wrapper(train_step):
    @wraps(train_step)
    def wrapper(*args, **kwargs):
        args_ = get_args()
        ret = train_step(*args, **kwargs)
        is_profile = args_.profile_npu and (
                (torch.distributed.get_rank() in args_.profile_ranks)
                or (-1 in args_.profile_ranks)
        )
        if is_profile:
            args_.prof.step()
        return ret

    return wrapper