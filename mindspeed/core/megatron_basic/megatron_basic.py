import time

import torch
from torch import _C
from torch_npu.npu import _lazy_call, device as device_ctx_manager


def _set_cuda_rng_state(new_state, device=-1):
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
            instance = torch.nn.LayerNorm(normalized_shape=hidden_size, eps=eps,)
        elif config.normalization == "RMSNorm":
            from mindspeed.core.fusions.fused_rms_norm import RMSNorm
            instance = RMSNorm(dim=hidden_size, eps=eps, sequence_parallel=config.sequence_parallel, config=config)
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance
