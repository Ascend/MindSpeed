# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

import time
from functools import wraps

import torch
from torch import _C
from torch_npu.npu import _lazy_call, device as device_ctx_manager

from mindspeed.core.tensor_parallel.tp_2d.group_api_2d import TPYCollectiveComm
from mindspeed.core.tensor_parallel.tp_2d.layernorm_2d import LayerNorm2D
from mindspeed.core.tensor_parallel.tp_2d.rms_norm_2d import RMSNorm2D


def _set_cuda_rng_state(new_state: torch.Tensor, device: int = -1, graph_safe: bool = False):
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.npu.default_generators[idx]

            # if graph capturing, set the rng state in a cudagraphable way
            if graph_safe:
                default_generator.graphsafe_set_state(new_state)
            else:
                default_generator.set_state(new_state)

    _lazy_call(cb)


def _compile_dependencies():
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print('> compiling dataset index builder ...')
        from megatron.core.datasets.utils import compile_helpers
        compile_helpers()
        print('>>> done with dataset index builder. Compilation time: {:.3f} '
              'seconds'.format(time.time() - start_time), flush=True)


class PTNorm:

    def __new__(cls, config, hidden_size: int, eps: float = 1e-5):
        if config.normalization == "LayerNorm":
            if getattr(config, "tp_2d", False):
                instance = LayerNorm2D(
                    hidden_size,
                    eps=eps,
                    last_dim_split_comm_intf=TPYCollectiveComm(),
                )
            else:
                instance = torch.nn.LayerNorm(normalized_shape=hidden_size, eps=eps,)
        elif config.normalization == "RMSNorm":
            if getattr(config, "tp_2d", False):
                instance = RMSNorm2D(
                    hidden_size,
                    eps=eps,
                    last_dim_split_comm_intf=TPYCollectiveComm(),
                )
                instance.use_fused_rmsnorm = False
            else:
                from mindspeed.core.fusions.fused_rms_norm import RMSNorm
                instance = RMSNorm(dim=hidden_size, eps=eps, sequence_parallel=config.sequence_parallel, config=config)
                instance.config.use_fused_rmsnorm = True
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance


def get_device_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        backend = torch.distributed.get_backend()
        local_rank = args[0]
        if backend == 'hccl':
            if local_rank is None:
                device = torch.device('cuda')
            else:
                device = torch.device(f'cuda:{local_rank}')
        else:
            device = func(*args, **kwargs)
        return device
    return wrapper


def get_device_arch_version():
    return 8


def preload_tensors(self, write_buckets, non_blocking=True):
    """
    Preloads tensors in `state_dict` to host memory via CPU memory.

    Args:
        write_buckets (List): List of `WriteBucket` objects that define what to
            save in a checkpoint.
        non_blocking (bool, optional): knob to enable pinned D2H memcpy. Default is True.
    """
    result = []

    for bucket in write_buckets:
        file_name, storage_key, (bytes_data, tensor_data) = bucket
        tensor_data = [
            (item, tensor.to("cpu", non_blocking=False) if not tensor.is_cpu else tensor.clone()) for item, tensor in tensor_data
        ]
        result.append((file_name, storage_key, (bytes_data, tensor_data)))
    if non_blocking:
        torch.cuda.synchronize()
    return result
