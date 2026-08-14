from typing import Dict

from megatron_adaptor.patches.patch_manager import Patch

_MEGATRON_TRAINING_AVAILABLE = None


def is_megatron_training_available():
    """
    Check if megatron.training module is available.

    Returns:
        bool: True if megatron.training is available, False otherwise.
    """
    global _MEGATRON_TRAINING_AVAILABLE
    if _MEGATRON_TRAINING_AVAILABLE is not None:
        return _MEGATRON_TRAINING_AVAILABLE

    try:
        import megatron.training  # noqa: F401

        _MEGATRON_TRAINING_AVAILABLE = True
    except ModuleNotFoundError:
        _MEGATRON_TRAINING_AVAILABLE = False

    return _MEGATRON_TRAINING_AVAILABLE


class MindSpeedPatchesManager:
    """Patch manager for MindSpeed-owned patches.

    Reuses MA's ``Patch`` class with an independent ``patches_info`` registry, enabling
    MindSpeed patches to be applied/removed layer by layer on top of MA patches.
    Intentionally does not inherit from MA's ``PatchesManager``.
    """

    patches_info: Dict[str, Patch] = {}

    @staticmethod
    def register_patch(orig_func_name, new_func=None, force_patch=False, create_dummy=False):
        """Patch registration method. When this method is executed, the patch does not take effect in real time.
        It takes effect only after the apply_patches method is invoked. Other details are as follows:

        1. If `orig_func_name` does not exist and create_dummy is set to True, a dummy function is created to ensure
        that the import is normal.
        2. If `orig_func_name` is not None, `orig_func_name` is replaced with `new_func`.
        3. If the `new_func` function name ends with `wrapper` or `decorator`, then `new_func` is decorated on
        `orig_func_name` as a decorator, and the decorator can be superimposed repeatedly.
        4. When force_patch=False, a function cannot be replaced repeatedly (but can be decorated repeatedly),
        otherwise the replacement is overwritten.
        """
        if orig_func_name not in MindSpeedPatchesManager.patches_info:
            MindSpeedPatchesManager.patches_info[orig_func_name] = Patch(orig_func_name, new_func, create_dummy)
        else:
            MindSpeedPatchesManager.patches_info.get(orig_func_name).set_patch_func(new_func, force_patch)

    @staticmethod
    def remove_wrappers(orig_func_name, wrappers_name, remove_check=True):
        """Remove wrapper registered in orig_func_name."""
        if orig_func_name not in MindSpeedPatchesManager.patches_info:
            raise ValueError('The function <{}> not exist.'.format(orig_func_name))
        patch = MindSpeedPatchesManager.patches_info.get(orig_func_name)
        wrappers_len = len(patch.wrappers)
        patch.remove_wrappers(wrappers_name)
        if remove_check and wrappers_len == len(patch.wrappers):
            raise RuntimeError('Remove wrappers has not remove anything.')

    @staticmethod
    def remove_patches():
        for patch in MindSpeedPatchesManager.patches_info.values():
            patch.remove_patch()
            patch.remove_wrappers()

    @staticmethod
    def apply_patches():
        for patch in MindSpeedPatchesManager.patches_info.values():
            patch.apply_patch()

    @staticmethod
    def get_patch(orig_func_name):
        return MindSpeedPatchesManager.patches_info.get(orig_func_name)
