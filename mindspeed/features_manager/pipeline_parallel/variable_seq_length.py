"""Define variable sequences length feature of pipeline parallel training.

Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""

from argparse import ArgumentParser, Namespace

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class VariableSequenceLengthFeature(MindSpeedFeature):
    """Variable sequence length feature of pipeline parallel training."""

    def __init__(
        self,
        feature_name: str = "variable-seq-lengths",
        optimization_level: int = 2,
    ):
        super().__init__(feature_name, optimization_level)
        self._var_seq_lengths = None  # Initialize the attribute

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            "--variable-seq-lengths",
            action="store_true",
            help="Supports variable sequence lengths across "
            "batches/microbatches. Set this if the data loader "
            "supports variable sequence length generation "
            "across batches/microbatches. Because of the additional "
            "communication overhead incurred during pipeline parallelism,"
            "it should not be set if the sequence length "
            "is constant during training. if sequence length is "
            "constant during training.",
        )

    def pre_validate_args(self, args: Namespace):
        self._var_seq_lengths = args.variable_seq_lengths
        if getattr(args, 'num_moe_experts', None) is None:
            args.variable_seq_lengths = False

    def post_validate_args(self, args: Namespace):
        args.variable_seq_lengths = self._var_seq_lengths

    def register_patches(
        self,
        patch_manager: MindSpeedPatchesManager,
        args: Namespace,
    ):
        # pylint: disable=import-outside-toplevel
        from mindspeed.core.pipeline_parallel.variable_seq_length.adaptor import (  # noqa
            mindspeed_communicate,
            mindspeed_commuticate_shapes,
        )

        if getattr(args, self.feature_name, None):
            patch_manager.register_patch(
                "megatron.core.pipeline_parallel.p2p_communication._communicate",  # noqa
                mindspeed_communicate,
            )
            patch_manager.register_patch(
                "megatron.core.pipeline_parallel.p2p_communication._communicate_shapes",  # noqa
                mindspeed_commuticate_shapes,
            )
