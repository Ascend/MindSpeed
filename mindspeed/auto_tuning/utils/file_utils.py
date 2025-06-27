from typing import Any, IO
from importlib import import_module
import os
import pickle
import stat

from mindspeed.auto_tuning.utils.mem_utils import mem_gb_to_b

_ALLOWED_MAX_FILE_SIZE = mem_gb_to_b(5)


class _RestrictedUnpickler(pickle.Unpickler):

    def find_class(self, module_name: str, global_name: str):
        if module_name.startswith("mindspeed.auto_tuning"):
            module = import_module(module_name)
            class_ = getattr(module, global_name)
            if isinstance(class_, type):
                return class_
        raise pickle.UnpicklingError(f"Global name '{module_name}.{global_name}' is forbidden.")


def check_file_size(file: IO):
    if os.fstat(file.fileno()).st_size > _ALLOWED_MAX_FILE_SIZE:
        raise IOError("File is too large to read.")


def restricted_read(file_path: str) -> Any:
    open_flags = os.O_RDONLY
    file_mode = stat.S_IWUSR | stat.S_IRUSR
    open_mode = "rb"
    encoding = None
    with os.fdopen(
        os.open(file_path, open_flags, mode=file_mode),
        mode=open_mode,
        encoding=encoding
    ) as file:
        check_file_size(file)
        return _RestrictedUnpickler(file).load()


def restricted_write(file_path: str, obj: object):
    open_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    file_mode = stat.S_IWUSR | stat.S_IRUSR
    open_mode = "wb"
    encoding = None
    with os.fdopen(
        os.open(file_path, open_flags, mode=file_mode),
        mode=open_mode,
        encoding=encoding
    ) as file:
        pickle.dump(obj, file)
