"""Define optimize send and recv communication feature
of pipeline parallel training.

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""

from argparse import ArgumentParser, Namespace

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class OptimizeSendRecvCommFeature(MindSpeedFeature):
    """ptimize send and recv communication feature
    of pipeline parallel training.
    """

    def __init__(
        self,
        feature_name: str = "optimize-send-recv-comm",
        optimization_level: int = 2,
    ):
        super().__init__(feature_name, optimization_level)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            "--optimize-send-recv-comm",
            action="store_true",
            help="optimize send_recv communication "
            "in pipeline without interleaving.",
        )

    def register_patches(
        self,
        patch_manager: MindSpeedPatchesManager,
        args: Namespace,
    ):
        # pylint: disable=import-outside-toplevel
        from mindspeed.core.pipeline_parallel.optimize_send_recv_comm.adaptor import \
            mindspeed_get_forward_backward_func  # noqa

        if getattr(args, self.feature_name, None):
            if (
                getattr(args, "num_layers_per_virtual_pipeline_stage", None)
                is None  # noqa
            ):
                patch_manager.register_patch(
                    "megatron.core.pipeline_parallel.schedules.get_forward_backward_func",  # noqa
                    mindspeed_get_forward_backward_func,
                )
