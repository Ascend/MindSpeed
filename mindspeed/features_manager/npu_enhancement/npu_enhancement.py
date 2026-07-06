# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""NpuEnhancementFeature: MindSpeed NPU enhancement patches.

This replaces the deleted megatron_basic/ features. It registers:
  - FP8: Fp8Recipe enum extension + get_fp8_context + TEDelayedScaling patches
  - Config: TransformerConfig init/post_init wrappers
  - Norm: PTNorm replacement
  - Bug fixes: duplicate allgather, count_zero, async save, weight_decay
  - Args: all MindSpeed-specific CLI arguments (migrated from deleted features)
"""

import importlib
import logging
import warnings

from mindspeed.features_manager.feature import MindSpeedFeature

logger = logging.getLogger("mindspeed.fp8.feature")


class NpuEnhancementFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('npu-enhancement', optimization_level=0)

    # ================================================================
    # Argument registration (migrated from RequirementsBasic + MegatronBasic + TransformerEngineBasic)
    # ================================================================
    def register_args(self, parser):
        # --- From RequirementsBasicFeature.register_args ---
        basic_group = parser.add_argument_group(title="mindspeed-basic")
        basic_group.add_argument(
            '--optimizer-selection',
            type=str,
            default='fused_adamw',
            choices=['fused_adamw', 'fused_torch_adamw', 'fused_ema_adamw'],
            help='Select from the former fused AdamW optimizer and Torch fused AdamW optimizer',
        )
        basic_group.add_argument(
            '--optimization-level',
            type=int,
            choices=[0, 1, 2],
            default=2,
            help='0: Minimum patch set, 1: Affinity optimization, 2: Advanced acceleration',
        )

        # --- From MegatronBasicFeature.register_args ---
        basic_group.add_argument("--use-fused-rmsnorm", action='store_true', help="Use fused rmsnorm.")
        basic_group.add_argument("--use-fused-swiglu", action='store_true', help="Use fused swiglu.")

        te_group = parser.add_argument_group(title="transformer-engine-basic")
        te_group.add_argument(
            '--no-use-gmm-fp8', action='store_false', help='not use GMM with scaling recipe.', dest='use_gmm_fp8'
        )
        te_group.add_argument(
            '--te-comparison-with-cpu',
            action='store_true',
            default=False,
            help='Compare the cast and quantmatmul of te on cpu and npu online.',
        )
        te_group.add_argument(
            '--te-comparison-with-bf16',
            action='store_true',
            default=False,
            help='Compare the cast and quantmatmul of te with bf16 online.',
        )
        te_group.add_argument(
            '--te-gmm-mode',
            choices=['performance', 'compatible'],
            default='compatible',
            help='Select the TE-GMM execution mode.',
            dest='te_gmm_mode',
        )
        te_group.add_argument(
            "--fp8-reuse-quantized-weight",
            action="store_true",
            default=False,
            help="Reuse quantized FP8 weight tensors within one optimizer step.",
        )

    # ================================================================
    # Argument validation (migrated from MegatronBasic + TransformerEngineBasic)
    # ================================================================
    def validate_args(self, args):
        # --- From MegatronBasicFeature.validate_args ---
        # Fix VPP when VPP_size=1 from megatron core_r0.14.0
        if (
            getattr(args, 'num_layers_per_virtual_pipeline_stage', None) is not None
            or getattr(args, 'num_virtual_stages_per_pipeline_rank', None) is not None
        ):
            if args.virtual_pipeline_model_parallel_size == 1 and not getattr(args, 'moe_fb_overlap', False):
                args.virtual_pipeline_model_parallel_size = None
                args.overlap_p2p_comm = False

        if (
            getattr(args, 'num_layers_per_virtual_pipeline_stage', None) is not None
            and getattr(args, 'pipeline_model_parallel_size', None) is not None
            and args.num_layers_per_virtual_pipeline_stage * args.pipeline_model_parallel_size == args.num_layers
        ):
            raise ValueError(
                'num_layers_per_virtual_pipeline_stage * pipeline_model_parallel_size == num_layers, '
                'please close --num-layers-per-virtual-pipeline-stage'
            )

        if getattr(args, 'defer_embedding_wgrad_compute', False):
            raise AssertionError(
                '--defer_embedding_wgrad_compute, although exclusive to TE scenarios, is not yet supported.'
            )

        # --- From TransformerEngineBasicFeature.validate_args ---
        if args.fp8 and args.transformer_impl == 'local':
            raise AssertionError('FP8 just support TE implement.')
        if args.use_ascend_coc and args.transformer_impl == 'transformer_engine':
            raise AssertionError('transformer engine does not support ascend coc')
        if args.use_ascend_mc2 and args.fp8 and args.fp8_recipe != 'mxfp8':
            raise AssertionError('MC2 is supported only by the mxfp8 recipe in fp8.')
        if getattr(args, "transformer_impl", "transformer_engine") == "transformer_engine" and getattr(
            args, "use_legacy_models", False
        ):
            raise AssertionError('transformer engine only support for mcore models')
        if args.fp8 == 'hif8':
            if args.fp8_recipe != 'tensorwise':
                raise ValueError("hif8 only support tensorwise scaling type")
        if args.use_gmm_fp8:
            if args.fp8_recipe not in ('mxfp8', 'tensorwise', 'delayed'):
                warnings.warn(
                    f"gmm fp8 only supports tensorwise, mxfp8, and delayed recipe, "
                    f"but {args.fp8_recipe} provided, using bf16 gmm instead."
                )
        if getattr(args, "fp8_reuse_quantized_weight", False) and not args.fp8:
            raise ValueError("fp8_reuse_quantized_weight is only valid when FP8 training is enabled")

    # ================================================================
    # Patch registration (migrated from MegatronBasic + TransformerEngineBasic)
    # ================================================================
    def pre_register_patches(self, patch_manager, args):
        """Note: Patches have been removed because they are
        already covered by MA (MegatronAdaptor)
        """
        pass

    def register_patches(self, patch_manager, args):
        # ================================================================
        # Step 1: Megatron config patches (migrated from MegatronBasicFeature)
        # ================================================================
        self._register_config_patches(patch_manager)

        # ================================================================
        # Step 2: GDN — replace GatedDeltaNet with MindSpeed subclass
        # (force_patch=True overrides MA's torch-native FLA operators
        #  in favour of MindSpeed Triton-accelerated implementations)
        # ================================================================
        self._register_gdn_patch(patch_manager)

        # ================================================================
        # Step 3: Norm patches (PTNorm)
        # ================================================================
        self._register_norm_patches(patch_manager)

        # ================================================================
        # Step 5: Bug fix patches (migrated from MegatronBasicFeature)
        # ================================================================
        self._register_bugfix_patches(patch_manager)

        # ================================================================
        # Step 7: Non-mcore patches (args parser + compile deps)
        # ================================================================
        self._register_non_mcore_patches(patch_manager)

    # ================================================================
    # Internal patch methods
    # ================================================================
    def _register_config_patches(self, patch_manager):
        """TransformerConfig patches for MindSpeed-specific parameters."""
        try:
            from mindspeed.core.megatron_basic.arguments_basic import (
                transformer_config_init_wrapper,
                transformer_config_post_init_wrapper,
            )

            patch_manager.register_patch(
                'megatron.core.transformer.transformer_config.TransformerConfig.__init__',
                transformer_config_init_wrapper,
                force_patch=True,
            )
            patch_manager.register_patch(
                'megatron.core.transformer.transformer_config.TransformerConfig.__post_init__',
                transformer_config_post_init_wrapper,
                force_patch=True,
            )
            logger.debug("TransformerConfig patches registered")
        except ImportError as e:
            logger.debug("TransformerConfig patches skipped: %s", e)

    def _register_gdn_patch(self, patch_manager):
        """Replace GatedDeltaNet with MindSpeed Triton-accelerated subclass.

        MA has already created dummy FLA modules so that Megatron's
        ``HAVE_FLA`` check passes.  We force_patch the entire class so
        that MindSpeed's subclass — which imports its own Triton operators
        directly — takes effect regardless of whether CP is enabled.
        """
        try:
            from mindspeed.core.ssm.gated_delta_net import GatedDeltaNet

            patch_manager.register_patch(
                'megatron.core.ssm.gated_delta_net.GatedDeltaNet', GatedDeltaNet, force_patch=True
            )
            logger.debug("GDN GatedDeltaNet patch registered (MindSpeed Triton)")
        except ImportError as e:
            logger.debug("GDN GatedDeltaNet patch skipped: %s", e)

    def _register_norm_patches(self, patch_manager):
        """PTNorm replacement for NPU fused RMSNorm."""
        try:
            from mindspeed.core.megatron_basic.megatron_basic import PTNorm

            for target in [
                'megatron.core.models.gpt.gpt_layer_specs.LNImpl',
                'megatron.core.transformer.torch_norm.WrappedTorchNorm',
                'megatron.core.transformer.transformer_block.LayerNormImpl',
                'megatron.core.extensions.transformer_engine.TENorm',
            ]:
                patch_manager.register_patch(target, PTNorm, force_patch=True)
            logger.debug("PTNorm patches registered")
        except ImportError as e:
            logger.debug("PTNorm patches skipped: %s", e)

    def _register_bugfix_patches(self, patch_manager):
        """Bug fix patches (migrated from MegatronBasicFeature)."""
        # Fix duplicate all-gather
        try:
            from mindspeed.core.optimizer.fix_duplicate_allgather import start_param_sync

            patch_manager.register_patch(
                'megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.start_param_sync',
                start_param_sync,
            )
            logger.debug("Bugfix: duplicate allgather registered")
        except ImportError:
            pass

        # Fix count_zeros in ChainedOptimizer
        try:
            from mindspeed.core.megatron_basic.count_zero_fix import step

            patch_manager.register_patch('megatron.core.optimizer.optimizer.ChainedOptimizer.step', step)
            logger.debug("Bugfix: count_zero registered")
        except ImportError:
            pass

        # Avoid async save issues
        try:
            from mindspeed.core.megatron_basic.megatron_basic import preload_tensors

            patch_manager.register_patch(
                'megatron.core.dist_checkpointing.strategies.filesystem_async.FileSystemWriterAsync.preload_tensors',
                preload_tensors,
            )
            logger.debug("Bugfix: async save registered")
        except ImportError:
            pass

        # Avoid incorrect weight_decay override in resume
        try:
            from mindspeed.core.megatron_basic.megatron_basic import dist_optim_load_state_dict

            patch_manager.register_patch(
                'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.load_state_dict',
                dist_optim_load_state_dict,
            )
            logger.debug("Bugfix: weight_decay override registered")
        except ImportError:
            pass

    def _register_non_mcore_patches(self, patch_manager):
        """Non-mcore arg parsing and init patches."""
        if importlib.util.find_spec("megatron.training") is None:
            return  # Mcore-only mode

        try:
            from mindspeed.core.megatron_basic.arguments_basic import (
                parse_args_wrapper,
                validate_args_wrapper,
                print_args_wrapper,
            )

            patch_manager.register_patch('megatron.training.arguments.parse_args', parse_args_wrapper)
            patch_manager.register_patch('megatron.training.arguments.validate_args', validate_args_wrapper)
            patch_manager.register_patch('megatron.training.arguments._print_args', print_args_wrapper)
            patch_manager.register_patch('megatron.training.yaml_arguments.validate_yaml', validate_args_wrapper)
            patch_manager.register_patch('megatron.training.yaml_arguments._print_args', print_args_wrapper)
            logger.debug("Non-mcore args patches registered")
        except ImportError as e:
            logger.debug("Non-mcore args patches skipped: %s", e)

        try:
            from mindspeed.core.megatron_basic.megatron_basic import _compile_dependencies, get_device_wrapper

            patch_manager.register_patch('megatron.training.initialize._compile_dependencies', _compile_dependencies)
            patch_manager.register_patch('megatron.training.dist_signal_handler.get_device', get_device_wrapper)
            from mindspeed.core.megatron_basic.megatron_basic import get_device_arch_version

            patch_manager.register_patch('megatron.training.utils.get_device_arch_version', get_device_arch_version)
            logger.debug("Non-mcore init patches registered")
        except ImportError as e:
            logger.debug("Non-mcore init patches skipped: %s", e)
