import sys
import inspect
from mindspeed.patch_utils import Patch


def clear_wrapper(aspm, original_func_name, target_func):
    reset_patch(aspm, original_func_name)
    orig_func = inspect.unwrap(target_func)
    apply_patch(original_func_name, orig_func)


def reset_patch(aspm, original_func_name):
    target_patch = aspm.patches_info[original_func_name]
    target_patch.wrappers = []


def apply_patch(original_func_name, new_func):

    split_name = original_func_name.rsplit('.', 1)
    if len(split_name) == 1:
        orig_module_name, orig_func_name = original_func_name, None
    else:
        orig_module_name, orig_func_name = split_name
    orig_module, orig_func = Patch.parse_path(orig_module_name, orig_func_name, False)
    final_patch_func = new_func
    if orig_func_name is not None:
        setattr(orig_module, orig_func_name, final_patch_func)
    for _, value in sys.modules.copy().items():
        if orig_func_name is not None and hasattr(value, orig_func_name) \
                and id(getattr(value, orig_func_name)) == id(orig_func):
            setattr(value, orig_func_name, final_patch_func)