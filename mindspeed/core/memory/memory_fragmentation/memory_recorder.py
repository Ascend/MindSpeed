from functools import wraps

from megatron.legacy.model.transformer import ParallelTransformer
from mindspeed.core.memory.memory_fragmentation.pluggable_allocator_adpator import load_memory_fragmentation_module


class MemoryRecorder(object):
    def __init__(self):
        load_memory_fragmentation_module().memory_recorder_start()

    def __del__(self):
        load_memory_fragmentation_module().memory_recorder_end()

    def register_recursive_hook(self, prefix_name, model):
        for name, module in model.named_children():
            if isinstance(module, ParallelTransformer):
                module.no_checkpoint_forward = module.forward
                module.forward = wrapper(module.forward)

            next_name = prefix_name + "." + name if prefix_name != "" else name
            self.register_recursive_hook(next_name, module)


def memory_recorder_wrapper(setup_model_and_optimizer):
    @wraps(setup_model_and_optimizer)
    def get_model_hook_func(*args, **kwargs):
        load_memory_fragmentation_module().precise_match_start()
        models, optimizer, lr_scheduler = setup_model_and_optimizer(*args, **kwargs)
        load_memory_fragmentation_module().precise_match_end()
        memory = MemoryRecorder()
        if isinstance(models, list):
            for model in models:
                memory.register_recursive_hook("module", model)
        else:
            memory.register_recursive_hook("module", models)
        return models, optimizer, lr_scheduler

    return get_model_hook_func


def wrapper(f):
    def rec_wrap(*args, **kwargs):
        recorder = MemoryRecorder()
        return f(*args, **kwargs)
    return rec_wrap
