from functools import wraps


def version_wrapper(fn):
    @wraps(fn)
    def wrapper(name, *args, **kwargs):
        return '2.2.0' if name == 'transformer-engine' else fn(name, *args, **kwargs)

    return wrapper


def dummy_compile(*args, **kwargs):
    if len(args) > 0 and callable(args[0]):

        def wrapper(*fn_args, **fn_kwargs):
            return args[0](*fn_args, **fn_kwargs)

        return wrapper
    else:

        def compile_wrapper(fn):
            def wrapper(*fn_args, **fn_kwargs):
                return fn(*fn_args, **fn_kwargs)

            return wrapper

        return compile_wrapper
