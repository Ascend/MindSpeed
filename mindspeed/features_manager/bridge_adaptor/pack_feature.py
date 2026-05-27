# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from argparse import ArgumentParser, Namespace

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class BridgePackFeature(MindSpeedFeature):

    def __init__(
            self
    ):
        super().__init__('pack-sequences-in-batch', 2)

    def register_patches(
            self,
            patch_manager: MindSpeedPatchesManager,
            args: Namespace,
    ):
        # pylint: disable=import-outside-toplevel
        from mindspeed.core.bridge.bridge_adaptor import get_tensor_shapes_in_megatron_bridge

        if getattr(args, self.feature_name, None):
            patch_manager.register_patch("megatron.core.pipeline_parallel.schedules.get_tensor_shapes",
                                         get_tensor_shapes_in_megatron_bridge)

            if getattr(args, 'mtp_num_layers', None):
                from mindspeed.core.bridge.bridge_adaptor import mtp_checkpointed_forward_impl

                patch_manager.register_patch(
                    "megatron.core.transformer.multi_token_prediction.MultiTokenPredictionLayer._checkpointed_forward",
                    mtp_checkpointed_forward_impl)
