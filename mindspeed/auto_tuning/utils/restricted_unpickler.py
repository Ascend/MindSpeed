import pickle
import importlib
from mindspeed.auto_tuning.utils.file_utils import check_file_size


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("mindspeed.auto_tuning"):
            module_ = importlib.import_module(module)
            clazz = getattr(module_, name)
            if isinstance(clazz, type):
                return clazz
        raise pickle.UnpicklingError("global '%s.%s' is forbidden" % (module, name))


def restricted_loads(s):
    check_file_size(s)
    return _RestrictedUnpickler(s).load()
