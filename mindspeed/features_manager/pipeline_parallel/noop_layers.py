"""Define noop layer feature of pipeline training."""

from argparse import ArgumentParser, Namespace
from logging import getLogger

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager

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
        if args.automated_pipeline:
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
        from mindspeed.core.pipeline_parallel.noop_layers.adaptor import (
            build_layers_adaptor, calc_flop_adaptor, track_moe_metrics_adaptor)

        if getattr(args, self.feature_name, None):
            patch_manager.register_patch(
                "megatron.core.transformer.transformer_block.TransformerBlock._build_layers",
                build_layers_adaptor,
            )
            patch_manager.register_patch(
                "megatron.training.training.num_floating_point_operations",
                calc_flop_adaptor,
            )
            patch_manager.register_patch(
                "megatron.core.transformer.moe.moe_utils.track_moe_metrics",
                track_moe_metrics_adaptor,
            )
