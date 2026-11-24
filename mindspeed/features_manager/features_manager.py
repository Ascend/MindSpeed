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

from megatron_adaptor.features_manager.features_manager import FeaturesManager


class MindSpeedFeaturesManager(FeaturesManager):
    """Feature manager that keeps MindSpeed patch ownership separate from MA.

    MindSpeed shares the unified feature list owned by ``FeaturesManager``:
    MindSpeed features are appended to it during the import of ``mindspeed.features_manager``,
    and the inherited apply/register/validate flows route patches via each feature's own ``patch_manager``.
    """

    @classmethod
    def remove_patches(cls):
        """Remove only the MindSpeed patch layer.

        MA patches stay applied so the MA implementation remains effective underneath;
        MindSpeed features are reset so they can be applied again (e.g. by ``repatch``).
        """
        from mindspeed.patch_utils import MindSpeedPatchesManager

        for feature in cls.FEATURES_LIST:
            if feature.patch_manager is MindSpeedPatchesManager:
                feature.pre_patches_applied = False
                feature.patches_applied = False
        MindSpeedPatchesManager.remove_patches()
