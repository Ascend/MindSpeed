"""Define noop layer feature of pipeline training.

Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""

from argparse import ArgumentParser, Namespace
from logging import getLogger

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import is_megatron_training_available, MindSpeedPatchesManager

LOG = getLogger(__name__)


class NoopLayersFeature(MindSpeedFeature):
    """Noop layers feature of pipeline parallel training."""

    def __init__(
        self,
        feature_name: str = "noop-layers",
        optimization_level: int = 2,
    ):
        super().__init__(feature_name, optimization_level)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            "--noop-layers",
            type=str,
            help="Specify the noop layers.",
        )

    def post_validate_args(self, args: Namespace):
        if getattr(args, "automated_pipeline", None):
            LOG.warning("disable noop_layers when enabling automated pipeline")
            args.noop_layers = None

        if isinstance(args.noop_layers, str):
            noop_layers = set()
            for x in args.noop_layers.split(","):
                if not x.isdigit():
                    raise ValueError(f"noop layer must be digit, but it's {x}")
                layer = int(x)
                if layer >= args.num_layers or layer < 0:
                    raise AssertionError(
                        f"each element in args.noop_layers({args.noop_layers})"
                        f" should bigger or equal to 0 "
                        f"and smaller than args.num_layers({args.num_layers})"
                    )
                noop_layers.add(layer)
            args.noop_layers = noop_layers

    def register_patches(
        self,
        patch_manager: MindSpeedPatchesManager,
        args: Namespace,
    ):
        # pylint: disable=import-outside-toplevel
        from mindspeed.core.pipeline_parallel.noop_layers.adaptor import (
            mindspeed_build_layers,
            mindspeed_calc_flop,
            mindspeed_track_moe_metrics,
        )

        if getattr(args, self.feature_name, None):
            patch_manager.register_patch(
                "megatron.core.transformer.transformer_block.TransformerBlock._build_layers",  # noqa
                mindspeed_build_layers,
            )

            megatron_training_available = is_megatron_training_available()
            if megatron_training_available:
                patch_manager.register_patch(
                    "megatron.training.training.num_floating_point_operations",
                    mindspeed_calc_flop,
                )

            patch_manager.register_patch(
                "megatron.core.transformer.moe.moe_utils.track_moe_metrics",
                mindspeed_track_moe_metrics,
            )
