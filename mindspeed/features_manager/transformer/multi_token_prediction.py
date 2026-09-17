# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from importlib import import_module

from mindspeed.features_manager.feature import MindSpeedFeature


class MultiTokenPredictionFeature(MindSpeedFeature):
    """Optimize native packed MTP movement independently of the pipeline schedule."""

    def __init__(self):
        # Reuse Megatron's argument; no separate optimization switch is needed.
        super().__init__('mtp-num-layers')

    def register_patches(self, patch_manager, args):
        if not getattr(args, 'mtp_num_layers', None):
            return

        module_name = 'megatron.core.transformer.multi_token_prediction'
        try:
            mtp_module = import_module(module_name)
        except ModuleNotFoundError as error:
            # Older cores may not provide MTP. Do not hide missing dependencies
            # inside an existing MTP module, which indicate a broken environment.
            if error.name and (error.name == module_name or module_name.startswith(error.name + '.')):
                return
            raise
        if not callable(getattr(mtp_module, '_roll_tensor_packed_seq', None)):
            return

        from mindspeed.core.transformer.multi_token_prediction import roll_tensor_packed_seq_wrapper

        patch_manager.register_patch(module_name + '._roll_tensor_packed_seq', roll_tensor_packed_seq_wrapper)
