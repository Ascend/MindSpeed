
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from argparse import ArgumentParser, Namespace

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import is_megatron_training_available, MindSpeedPatchesManager


class OptimizeP2PCommFeature(MindSpeedFeature):
    # Use isend/irecv instead of batch_isend_irecv on NPU for better performance for pipeline parallel.

    def __init__(
        self,
        feature_name: str = "pipeline-model-parallel-size",
        optimization_level: int = 2,
    ):
        super().__init__(feature_name, optimization_level)


    def register_patches(
        self,
        patch_manager: MindSpeedPatchesManager,
        args: Namespace,
    ):
        # pylint: disable=import-outside-toplevel
        from mindspeed.core.pipeline_parallel.optimize_p2p_comm.adaptor import (
            core_transformer_config_from_args_wrapper,
        )

        if getattr(args, self.feature_name, None) and int(args.pipeline_model_parallel_size) >= 2:
            if (
                getattr(args, "num_layers_per_virtual_pipeline_stage", None)
                is None  # noqa
            ):
                megatron_training_available = is_megatron_training_available()
                if megatron_training_available:
                    patch_manager.register_patch(
                        "megatron.training.arguments.core_transformer_config_from_args",  # noqa
                        core_transformer_config_from_args_wrapper,
                    )

