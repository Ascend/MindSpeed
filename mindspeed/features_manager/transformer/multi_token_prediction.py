# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
