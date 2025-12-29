from functools import wraps


def validate_global_plan_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return True
    return wrapper


def validate_non_overlapping_shards_metadata_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return
    return wrapper


def save_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        args_list = list(args)
        if len(args_list) > 5:
            args_list[4] = False
        if 'validate_access_integrity' in kwargs:
            kwargs['validate_access_integrity'] = False
        args = tuple(args_list)
        return func(*args, **kwargs)
    return wrapper