import os
from functools import wraps

from megatron import get_args
from megatron.core.parallel_state import get_tensor_model_parallel_rank

sizes_cache = {}


def build_key_sizes_wrapper(fn):
    @wraps(fn)
    def wrapper(key, data, *args, **kwargs):
        global sizes_cache
        if sizes_cache:
            if get_tensor_model_parallel_rank() == 0:
                message = f"Can not use this method on unfixed length datasets. " \
                 f"Please delete the arugument '--key-size-cache'."
                assert data[key[0]].size() == sizes_cache.get('ori_sizes', None), message

            return sizes_cache.get('key_sizes', None)

        key_size, key_numel, total_numel = fn(key, data, *args, **kwargs)
        
        args = get_args()
        if args.key_sizes_cache:
            if get_tensor_model_parallel_rank() == 0:
                print('> use sizes cache')
                sizes_cache['ori_sizes'] = data[key[0]].size()
            sizes_cache['key_sizes'] = [key_size, key_numel, total_numel]
            
        return key_size, key_numel, total_numel

    return wrapper
