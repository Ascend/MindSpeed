import torch_npu
from functools import wraps
from mindspeed.core.memory.memory_fragmentation.pluggable_allocator_adpator import load_memory_fragmentation_module

is_optimizer_init_end = False


def optimizer_init_wrapper(step):
    @wraps(step)
    def rec_wrap(*args, **kwargs):
        global is_optimizer_init_end            
        if not is_optimizer_init_end:
            torch_npu.npu.empty_cache()
            load_memory_fragmentation_module().precise_match_start()
        optimizer_initialized, grad_norm, num_zeros_in_grad = step(*args, **kwargs)
        if not is_optimizer_init_end:
            load_memory_fragmentation_module().precise_match_end()
        is_optimizer_init_end = optimizer_initialized

        return optimizer_initialized, grad_norm, num_zeros_in_grad

    return rec_wrap
