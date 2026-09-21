# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""Default-on deferred forward-send waits for the Megatron 0.18 interleaved schedule."""

from argparse import ArgumentParser, BooleanOptionalAction, Namespace

from mindspeed.features_manager.feature import MindSpeedFeature


class DeferP2PSendWaitFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__("defer-p2p-send-wait", optimization_level=2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            "--defer-p2p-send-wait",
            action=BooleanOptionalAction,
            default=True,
            help=(
                "Defer interleaved forward P2P send waits while retaining the original send storage (default: enabled). "
                "Automatically disabled for unsupported configurations; use --no-defer-p2p-send-wait to opt out."
            ),
        )

    def validate_args(self, args: Namespace):
        if not getattr(args, self.feature_name, False):
            return
        # These features replace the native interleaved schedule or its P2P API.
        incompatible_features = (
            "moe_fb_overlap",
            "use_multiparameter_pipeline_model_parallel",
            "recompute_in_bubble",
            "recompute_in_advance",
            "tp_2d",
            "variable_seq_lengths",
            "optimize_send_recv_comm",
            "dist_train",
        )
        # Full validation runs after Megatron/layout-derived PP/VP sizes exist.
        # Tolerate numeric strings for callers supplying their own Namespace.
        try:
            pp_size = int(getattr(args, "pipeline_model_parallel_size", None) or 0)
            vp_size = int(getattr(args, "virtual_pipeline_model_parallel_size", None) or 0)
        except (TypeError, ValueError, OverflowError):
            setattr(args, self.feature_name, False)
            return
        supported = (
            pp_size > 1
            and vp_size > 0
            and getattr(args, "overlap_p2p_comm", False)
            and not getattr(args, "batch_p2p_comm", False)
            and getattr(args, "deallocate_pipeline_outputs", True)
            and getattr(args, "schedules_method", None) is None
            and not any(getattr(args, name, False) for name in incompatible_features)
        )
        if not supported:
            setattr(args, self.feature_name, False)

    def register_patches(self, patch_manager, args: Namespace):
        # Registration uses early adaptor args: PP can still be a string and VP
        # may not exist until Megatron derives it. Only honor the explicit switch
        # here; the schedule wrapper checks the final validated switch at runtime.
        if not getattr(args, self.feature_name, False):
            return
        from mindspeed.core.pipeline_parallel.defer_p2p_send_wait import (
            forward_backward_pipelining_with_interleaving_wrapper,
            send_forward_recv_forward_wrapper,
        )

        patch_manager.register_patch(
            "megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving",
            forward_backward_pipelining_with_interleaving_wrapper,
        )
        patch_manager.register_patch(
            "megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.send_forward_recv_forward",
            send_forward_recv_forward_wrapper,
        )
