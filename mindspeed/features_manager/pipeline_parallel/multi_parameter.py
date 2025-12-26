"""Define multi parameter feature of pipeline parallel training.

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""

from argparse import ArgumentParser, Namespace

import torch

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class MultiParameterFeature(MindSpeedFeature):
    """Multi parameter feature of pipeline parallel training."""

    def __init__(
        self,
        feature_name: str = "use-multiparameter-pipeline-model-parallel",
        optimization_level: int = 2,
    ):
        super().__init__(feature_name, optimization_level)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            "--use-multiparameter-pipeline-model-parallel",
            action="store_true",
            default=False,
            help="can transfer multi parameters from "
            "stage to stage in pipeline model parallel",
        )

    def validate_args(self, args):
        self.incompatible_check(args, 'moe_fb_overlap')

        if getattr(args, "use_multiparameter_pipeline_model_parallel", False) and (not hasattr(args, 'pipeline_tensor_shapes') or args.pipeline_tensor_shapes is None):
            if getattr(args, "schedules_method", False) == "dualpipev":
                raise AssertionError(
                    "The dualpipev and use_multiparameter_pipeline_model_parallel are incompatible."
                )
            tensor_shape = (int(args.seq_length / args.context_parallel_size), args.micro_batch_size, args.hidden_size)

            if getattr(args, "bf16", False):
                dtype = torch.bfloat16
            elif getattr(args, "fp16", False):
                dtype = torch.float16
            else:
                dtype = torch.float32

            args.pipeline_tensor_shapes = [{"shape": tensor_shape, "dtype": dtype}]

    def register_patches(
        self,
        patch_manager: MindSpeedPatchesManager,
        args: Namespace,
    ):
        # pylint: disable=import-outside-toplevel
        from mindspeed.core.pipeline_parallel.multi_parameter.adaptor import (
            get_tensor_shapes_wrapper,
            forward_step_wrapper,
            mindspeed_backward_step,
            mindspeed_recv_forward,
            mindspeed_recv_backward,
            mindspeed_send_forward,
            mindspeed_send_backward,
            mindspeed_send_forward_recv_backward,
            mindspeed_send_backward_recv_forward,
            get_forward_backward_func_wrapper,
            core_transformer_config_from_args_wrapper,
        )

        if getattr(args, self.feature_name, None):
            patch_manager.register_patch(
                "megatron.core.pipeline_parallel.schedules.get_tensor_shapes",
                get_tensor_shapes_wrapper,
            )
            patch_manager.register_patch(
                "megatron.core.pipeline_parallel.schedules.forward_step",
                forward_step_wrapper,
            )
            patch_manager.register_patch(
                "megatron.core.pipeline_parallel.schedules.backward_step",
                mindspeed_backward_step,
            )
            patch_manager.register_patch(
                "megatron.core.pipeline_parallel.schedules.recv_forward",
                mindspeed_recv_forward,
            )
            patch_manager.register_patch(
                "megatron.core.pipeline_parallel.schedules.recv_backward",
                mindspeed_recv_backward,
            )
            patch_manager.register_patch(
                "megatron.core.pipeline_parallel.schedules.send_forward",
                mindspeed_send_forward,
            )
            patch_manager.register_patch(
                "megatron.core.pipeline_parallel.schedules.send_backward",
                mindspeed_send_backward,
            )
            patch_manager.register_patch(
                "megatron.core.pipeline_parallel.schedules.send_forward_recv_backward",  # noqa
                mindspeed_send_forward_recv_backward,
            )
            patch_manager.register_patch(
                "megatron.core.pipeline_parallel.schedules.send_backward_recv_forward",  # noqa
                mindspeed_send_backward_recv_forward,
            )
            patch_manager.register_patch(
                "megatron.core.pipeline_parallel.schedules.get_forward_backward_func",  # noqa
                get_forward_backward_func_wrapper,
            )
            patch_manager.register_patch(
                "megatron.training.arguments.core_transformer_config_from_args",  # noqa
                core_transformer_config_from_args_wrapper,
            )
