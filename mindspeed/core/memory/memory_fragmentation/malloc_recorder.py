from functools import wraps
from mindspeed.core.memory.memory_fragmentation.pluggable_allocator_adpator import load_memory_fragmentation_module


class MallocRecorder(object):
    def __init__(self):
        load_memory_fragmentation_module().malloc_recorder_start()

    def __del__(self):
        load_memory_fragmentation_module().malloc_recorder_end()


def malloc_recorder_wrapper(train_step):
    @wraps(train_step)
    def rec_wrap(*args, **kwargs):
        recorder = MallocRecorder()
        return train_step(*args, **kwargs)
    return rec_wrap
