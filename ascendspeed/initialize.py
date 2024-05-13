# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import time
from functools import wraps
import torch
from megatron.initialize import _warmup_jit_function
from ascendspeed.core.tensor_parallel.ascend_turbo.initialize import initialize_cfg_from_args


def _compile_dependencies():
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print('> compiling dataset index builder ...')
        from megatron.core.datasets.utils import compile_helpers
        compile_helpers()
        print('>>> done with dataset index builder. Compilation time: {:.3f} '
              'seconds'.format(time.time() - start_time), flush=True)


def set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
        # nvfuser
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        # torch._C._jit_set_nvfuser_enabled(True)
        torch._C._debug_set_autodiff_subgraph_inlining(False)
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)

    _warmup_jit_function()


def coc_registration_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        from megatron import get_args
        from ascendspeed.core.tensor_parallel.lcal_coc.user_config import initialize_coc_from_cfg
        args = get_args()
        initialize_coc_from_cfg(args)
        return res

    return wrapper


def mc2_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        from megatron import get_args
        args = get_args()
        initialize_cfg_from_args(args)
        return res

    return wrapper
