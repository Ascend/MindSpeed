from functools import wraps
from megatron import get_args
from megatron.core import mpu
from ascendspeed.core.context_parallel.ulysses_context_parallel import UlyssesContextAttention


def attention_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *arg, **kwargs):
        fn(self, *arg, **kwargs)

        args = get_args()
        if args.context_parallel_size > 1 and args.context_parallel_algo == 'ulysses_cp_algo':
            self.core_attention = UlyssesContextAttention(self.core_attention, mpu.get_context_parallel_group())

    return wrapper
