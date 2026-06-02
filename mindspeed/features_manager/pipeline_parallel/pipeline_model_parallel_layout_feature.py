# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from argparse import ArgumentParser, Namespace

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class PipelineModelParallelLayoutFeature(MindSpeedFeature):
    """Support --pipeline-model-parallel-layout from Megatron dev on core_r0.12.1."""

    _STATE_ATTR = "_mindspeed_pipeline_model_parallel_layout_state"

    def __init__(
        self,
        feature_name: str = "pipeline-model-parallel-layout",
        optimization_level: int = 0,
    ):
        super().__init__(feature_name, optimization_level)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title="pipeline-model-parallel-layout")
        group.add_argument(
            "--pipeline-model-parallel-layout",
            type=str,
            default=None,
            help=(
                "A string that describes a custom pipeline model parallel layout. "
                'e.g., "E|(t|)*3,m|m||L". E, L, t, m denotes embedding, loss, transformer '
                'decoder layer, and mtp layer, respectively. Stages are split by "|". '
                "Replicated stages or layers can be described with multiplication. "
                "Commas can be used cosmetically. "
                "Default None is not using this argument to set the layout."
            ),
        )

    def pre_validate_args(self, args: Namespace):
        if (
            getattr(args, self.feature_name, None) is None
            or not isinstance(args, Namespace)
            or not hasattr(args, "pipeline_model_parallel_size")
        ):
            return

        setattr(
            args,
            self._STATE_ATTR,
            {
                "num_layers_per_virtual_pipeline_stage": getattr(args, "num_layers_per_virtual_pipeline_stage", None),
                "num_virtual_stages_per_pipeline_rank": getattr(args, "num_virtual_stages_per_pipeline_rank", None),
                "decoder_first_pipeline_num_layers": getattr(args, "decoder_first_pipeline_num_layers", None),
                "decoder_last_pipeline_num_layers": getattr(args, "decoder_last_pipeline_num_layers", None),
                "overlap_p2p_comm": getattr(args, "overlap_p2p_comm", None),
                "align_param_gather": getattr(args, "align_param_gather", None),
            },
        )

        args.num_layers_per_virtual_pipeline_stage = None
        from mindspeed.core.pipeline_parallel.pipeline_model_parallel_layout.layout import (
            PipelineParallelLayerLayout,
        )

        num_stages = PipelineParallelLayerLayout.get_num_stages_from_str(args.pipeline_model_parallel_layout)
        detected_vpp_size = num_stages // args.pipeline_model_parallel_size
        if detected_vpp_size > 1:
            args.num_virtual_stages_per_pipeline_rank = detected_vpp_size
        else:
            args.num_virtual_stages_per_pipeline_rank = None

            # Megatron core_r0.12.1 still enforces even num_layers / pp_size when it does
            # not see an uneven-pipeline option. The custom layout will validate the real
            # partition after Megatron validation, so use a temporary uneven marker here.
            if (
                getattr(args, "decoder_first_pipeline_num_layers", None) is None
                and getattr(args, "decoder_last_pipeline_num_layers", None) is None
            ):
                args.decoder_first_pipeline_num_layers = 1

    def post_validate_args(self, args: Namespace):
        state = getattr(args, self._STATE_ATTR, None)
        if state is None or not isinstance(args, Namespace) or not hasattr(args, "pipeline_model_parallel_size"):
            return

        for key, value in state.items():
            if key in (
                "num_layers_per_virtual_pipeline_stage",
                "num_virtual_stages_per_pipeline_rank",
                "overlap_p2p_comm",
                "align_param_gather",
            ):
                continue
            setattr(args, key, value)

    def validate_args(self, args: Namespace):
        if (
            getattr(args, self.feature_name, None) is None
            or not isinstance(args, Namespace)
            or not hasattr(args, "pipeline_model_parallel_size")
        ):
            return

        if getattr(args, "schedules_method", None) == "dualpipev":
            raise AssertionError("--pipeline-model-parallel-layout is incompatible with --schedules-method dualpipev.")
        if getattr(args, "pipeline_num_transformer_layers", None) is not None:
            raise AssertionError(
                "--pipeline-model-parallel-layout is incompatible with --pipeline-num-transformer-layers."
            )
        if getattr(args, "noop_layers", None) is not None:
            raise AssertionError("--pipeline-model-parallel-layout is incompatible with --noop-layers.")
        if getattr(args, "recompute_in_bubble", False) or getattr(args, "recompute_in_advance", False):
            raise AssertionError(
                "--pipeline-model-parallel-layout is not supported with "
                "--recompute-in-bubble or --recompute-in-advance now."
            )

        state = getattr(args, self._STATE_ATTR, {})
        num_layers_per_virtual_pipeline_stage = state.get(
            "num_layers_per_virtual_pipeline_stage",
            getattr(args, "num_layers_per_virtual_pipeline_stage", None),
        )
        num_virtual_stages_per_pipeline_rank = state.get(
            "num_virtual_stages_per_pipeline_rank",
            getattr(args, "num_virtual_stages_per_pipeline_rank", None),
        )

        assert (
            int(num_layers_per_virtual_pipeline_stage is not None)
            + int(num_virtual_stages_per_pipeline_rank is not None)
            + int(args.pipeline_model_parallel_layout is not None)
        ) <= 1, (
            "No more than one of the following arguments can be set at the same time: "
            "--num-layers-per-virtual-pipeline-stage, --num-virtual-stages-per-pipeline-rank,"
            "--pipeline-model-parallel-layout. "
            f"{num_layers_per_virtual_pipeline_stage=}, "
            f"{num_virtual_stages_per_pipeline_rank=}, "
            f"{args.pipeline_model_parallel_layout=}."
        )
        if state:
            args.num_layers_per_virtual_pipeline_stage = num_layers_per_virtual_pipeline_stage
            args.num_virtual_stages_per_pipeline_rank = num_virtual_stages_per_pipeline_rank

        from mindspeed.core.pipeline_parallel.pipeline_model_parallel_layout.layout import (
            PipelineParallelLayerLayout,
        )

        num_stages = PipelineParallelLayerLayout.get_num_stages_from_str(args.pipeline_model_parallel_layout)
        assert num_stages % args.pipeline_model_parallel_size == 0, (
            f"The length of pipeline_model_parallel_layout must be divisible"
            f" by pipeline_model_parallel_size ({num_stages=},"
            f" {args.pipeline_model_parallel_size=})"
        )

        args.virtual_pipeline_model_parallel_size = num_stages // args.pipeline_model_parallel_size
        if args.virtual_pipeline_model_parallel_size == 1:
            args.virtual_pipeline_model_parallel_size = None
        elif getattr(args, "optimize_send_recv_comm", False):
            raise AssertionError(
                "--pipeline-model-parallel-layout with virtual pipeline is incompatible with --optimize-send-recv-comm."
            )

        original_overlap_p2p_comm = state.get("overlap_p2p_comm", getattr(args, "overlap_p2p_comm", None))
        original_align_param_gather = state.get("align_param_gather", getattr(args, "align_param_gather", None))
        if original_overlap_p2p_comm is not None:
            args.overlap_p2p_comm = original_overlap_p2p_comm
        if original_align_param_gather is not None:
            args.align_param_gather = original_align_param_gather

        if args.virtual_pipeline_model_parallel_size is not None:
            if args.overlap_p2p_comm:
                assert args.pipeline_model_parallel_size > 1, (
                    "When interleaved schedule is used, pipeline-model-parallel size should be greater than 1"
                )
            else:
                assert args.pipeline_model_parallel_size > 2, (
                    "When interleaved schedule is used and p2p communication overlap is disabled, "
                    "pipeline-model-parallel size should be greater than 2 to avoid having multiple "
                    "p2p sends and recvs between same 2 ranks per communication batch"
                )
        else:
            args.overlap_p2p_comm = False
            args.align_param_gather = False

    def register_patches(
        self,
        patch_manager: MindSpeedPatchesManager,
        args: Namespace,
    ):
        if not getattr(args, self.feature_name, None):
            return

        from mindspeed.core.pipeline_parallel.pipeline_model_parallel_layout.adaptor import (
            LayerType,
            get_num_layers_to_build_wrapper,
            get_transformer_layer_offset_wrapper,
            transformer_config_post_init_wrapper,
        )

        patch_manager.register_patch("megatron.core.transformer.enums.LayerType", LayerType)
        patch_manager.register_patch(
            "megatron.core.transformer.transformer_config.TransformerConfig.__post_init__",
            transformer_config_post_init_wrapper,
        )
        patch_manager.register_patch(
            "megatron.core.transformer.transformer_block.get_num_layers_to_build",
            get_num_layers_to_build_wrapper,
        )
        patch_manager.register_patch(
            "megatron.core.transformer.transformer_layer.get_transformer_layer_offset",
            get_transformer_layer_offset_wrapper,
        )
