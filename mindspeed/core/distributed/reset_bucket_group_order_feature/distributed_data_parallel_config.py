from functools import wraps
from megatron.training import get_args


def distributed_data_parallel_config_init_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn(*args, **kwargs)
        self = args[0]
        global_args = get_args()
        reset_bucket_group_order = getattr(global_args, "reset_bucket_group_order", False)
        setattr(self, "reset_bucket_group_order", reset_bucket_group_order)
    return wrapper