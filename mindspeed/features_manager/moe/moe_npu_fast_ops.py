# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser
import warnings

from mindspeed.features_manager.feature import MindSpeedFeature


def fast_ops_conflicts(args):
    # These features replace the native staged MoE API. Capacity padding also
    # requires indexed probability selection, not the dropless masked_select.
    flags = (
        "moe_fb_overlap",
        "moe_alltoall_overlap_comm",
        "moe_allgather_overlap_comm",
        "moe_alltoall_mc2",
        "moe_tp_extend_ep",
        "moe_pad_expert_input_to_capacity",
        "enable_cuda_graph",
        "external_cuda_graph",
    )
    conflicts = [name for name in flags if getattr(args, name, False)]
    if getattr(args, "cuda_graph_impl", None) not in (None, "none"):
        conflicts.append("cuda_graph_impl")
    return conflicts


class MoENpuFastOpsFeature(MindSpeedFeature):
    """Default-on NPU MoE ops and eager dispatch temporary lifetime optimization.

    Restores the mcore 0.12 op selection (``masked_select`` permute, ``scatter``
    routing map, ``scatter_add_`` unpermute) to avoid the AICPU ``aclnnSort`` /
    ``aclnnIndexPutImpl`` kernels that Megatron 0.15+/0.17+ selects on NPU. The
    routing patch stays active with ``--moe-permute-fusion`` (whose fused NPU
    operators replace permute/unpermute themselves), and the whole feature can be
    disabled with ``--no-moe-npu-fast-ops``. The same switch controls the eager
    forward optimization that releases unused dispatch temporaries before experts.
    """

    def __init__(self):
        super().__init__("moe-npu-fast-ops", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            "--no-moe-npu-fast-ops",
            action="store_false",
            dest="moe_npu_fast_ops",
            default=True,
            help="Disable the NPU fast-path MoE ops (masked_select permute, scatter-based routing "
            "map, scatter_add_ unpermute) and eager dispatch memory optimization. "
            "Automatically disabled for incompatible MoE or CUDA Graph configurations.",
        )

    def is_need_apply(self, args):
        return getattr(args, self.feature_name, True)

    def validate_args(self, args):
        if not self.is_need_apply(args):
            return
        conflicts = fast_ops_conflicts(args)
        if conflicts:
            setattr(args, self.feature_name, False)
            warnings.warn(
                "Disabling moe-npu-fast-ops (including dispatch memory optimization) "
                "for incompatible configuration: " + ", ".join(conflicts) + ". Using the existing MoE implementation."
            )

    def register_patches(self, patch_manager, args):
        # Avoid wrapping replacement MoE classes during early patch application.
        # Full validation also turns off the switch for the final argument set.
        if not self.is_need_apply(args) or fast_ops_conflicts(args):
            return
        from mindspeed.core.transformer.moe.moe_npu_fast_ops import (
            moe_forward_dispatch_lifetime_wrapper,
            permute_npu_wrapper,
            topk_routing_with_score_function_scatter_wrapper,
            unpermute_npu_wrapper,
        )

        patch_manager.register_patch(
            "megatron.core.transformer.moe.moe_utils.topk_routing_with_score_function",
            topk_routing_with_score_function_scatter_wrapper,
        )

        patch_manager.register_patch(
            "megatron.core.transformer.moe.moe_layer.MoELayer.forward",
            moe_forward_dispatch_lifetime_wrapper,
        )

        if getattr(args, "moe_permute_fusion", False) or getattr(
            args, "use_fused_moe_token_permute_and_unpermute", False
        ):
            # FusedMoEPermuteFeature replaces permute/unpermute with fused NPU
            # operators, so its patches own those two targets.
            return

        patch_manager.register_patch("megatron.core.transformer.moe.moe_utils.permute", permute_npu_wrapper)
        patch_manager.register_patch("megatron.core.transformer.moe.moe_utils.unpermute", unpermute_npu_wrapper)
