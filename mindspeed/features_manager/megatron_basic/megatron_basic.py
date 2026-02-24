import sys

from mindspeed.features_manager.feature import MindSpeedFeature


def _repair_modelopt_torch_tree():
    try:
        mo = sys.modules.get("modelopt")
        if mo is None:
            return
        if hasattr(mo, "torch"):
            return

        for k in list(sys.modules.keys()):
            if k == "modelopt" or k == "modelopt.torch" or k.startswith("modelopt.torch."):
                sys.modules.pop(k, None)
    except Exception:
        pass


class MegatronBasicFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('megatron-basic', optimization_level=0)

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--use-fused-rmsnorm", action='store_true', help="Use fused rmsnorm.")
        group.add_argument("--use-fused-swiglu", action='store_true', help="Use fused swiglu.")

    def validate_args(self, args):
        # Fix VPP when VPP_size=1 from megatron core_r0.14.0 (!3640).
        if getattr(args, 'num_layers_per_virtual_pipeline_stage', None) is not None or getattr(args, 'num_virtual_stages_per_pipeline_rank', None) is not None:
            if args.virtual_pipeline_model_parallel_size == 1 and not getattr(args, 'moe_fb_overlap', False):
                args.virtual_pipeline_model_parallel_size = None
                args.overlap_p2p_comm = False
        
        if (getattr(args, 'num_layers_per_virtual_pipeline_stage', None) is not None and 
            getattr(args, 'pipeline_model_parallel_size', None) is not None and 
            args.num_layers_per_virtual_pipeline_stage * args.pipeline_model_parallel_size == args.num_layers):
            raise ValueError(
                'num_layers_per_virtual_pipeline_stage * pipeline_model_parallel_size == num_layers, '
                'please close --num-layers-per-virtual-pipeline-stage'
            )
        
        if getattr(args, 'defer_embedding_wgrad_compute', False):
            raise AssertionError(
                '--defer_embedding_wgrad_compute, although exclusive to TE scenarios, is not yet supported.'
            )

    def register_patches(self, patch_manager, args):
        try:
            import megatron.training
            _repair_modelopt_torch_tree()
            only_mcore = False
        except ModuleNotFoundError:
            only_mcore = True

        self.register_mcore_basic_patches(patch_manager, args)
        if not only_mcore:
            self.register_non_mcore_basic_patches(patch_manager, args)

    def register_mcore_basic_patches(self, pm, args):
        # configuration patches
        from mindspeed.core.megatron_basic.arguments_basic import transformer_config_init_wrapper, transformer_config_post_init_wrapper
        pm.register_patch("megatron.core.transformer.transformer_config.TransformerConfig.__init__", transformer_config_init_wrapper)
        pm.register_patch("megatron.core.transformer.transformer_config.MLATransformerConfig.__init__", transformer_config_init_wrapper)
        pm.register_patch("megatron.core.transformer.transformer_config.TransformerConfig.__post_init__", transformer_config_post_init_wrapper)

        # initialization patches
        from mindspeed.core.megatron_basic.megatron_basic import _set_cuda_rng_state
        pm.register_patch('megatron.core.tensor_parallel.random._set_cuda_rng_state', _set_cuda_rng_state)

        # norm patches
        from mindspeed.core.megatron_basic.megatron_basic import PTNorm
        pm.register_patch('megatron.core.models.gpt.gpt_layer_specs.LNImpl', PTNorm)
        pm.register_patch('megatron.core.transformer.torch_norm.WrappedTorchNorm', PTNorm)
        pm.register_patch('megatron.core.transformer.transformer_block.LayerNormImpl', PTNorm)
        pm.register_patch('megatron.core.extensions.transformer_engine.TENorm', PTNorm)

        # coalescing_manager patches
        from mindspeed.core.distributed.param_and_grad_buffer import start_param_sync, finish_param_sync, start_grad_sync, finish_grad_sync
        pm.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.start_param_sync', start_param_sync)
        pm.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.finish_param_sync', finish_param_sync)
        pm.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.start_grad_sync', start_grad_sync)
        pm.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.finish_grad_sync', finish_grad_sync)

        # fix duplicate all-gather
        from mindspeed.core.optimizer.fix_duplicate_allgather import start_param_sync
        from mindspeed.core.optimizer.fix_duplicate_allgather import step_with_ready_grads_distrib_opti_wrapper
        from mindspeed.core.optimizer.fix_duplicate_allgather import get_megatron_optimizer_wrapper
        pm.register_patch('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.start_param_sync', start_param_sync)
        pm.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.step_with_ready_grads', step_with_ready_grads_distrib_opti_wrapper)
        pm.register_patch('megatron.core.optimizer.get_megatron_optimizer', get_megatron_optimizer_wrapper)

        # Currently, it is not supported to Cast shard fp32 main params to fp8 model params
        from mindspeed.core.fp8_utils import quantize_param_shard
        pm.register_patch('megatron.core.fp8_utils.quantize_param_shard', quantize_param_shard)

        # fix count_zeros in ChainedOptimizer for core_r0.12.1.
        from mindspeed.core.megatron_basic.count_zero_fix import step
        pm.register_patch('megatron.core.optimizer.optimizer.ChainedOptimizer.step', step)

        # avoid async save
        from mindspeed.core.megatron_basic.megatron_basic import preload_tensors
        pm.register_patch('megatron.core.dist_checkpointing.strategies.filesystem_async.FileSystemWriterAsync.preload_tensors', preload_tensors)

        # avoid incorrect weight_decay override in resume task
        from mindspeed.core.megatron_basic.megatron_basic import dist_optim_load_state_dict
        pm.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.load_state_dict', dist_optim_load_state_dict)

    def register_non_mcore_basic_patches(self, pm, args):
        # args parser patch
        from mindspeed.core.megatron_basic.arguments_basic import parse_args_wrapper, validate_args_wrapper, print_args_wrapper
        pm.register_patch('megatron.training.arguments.parse_args', parse_args_wrapper)
        pm.register_patch('megatron.training.arguments.validate_args', validate_args_wrapper)
        pm.register_patch('megatron.training.arguments._print_args', print_args_wrapper)
        pm.register_patch('megatron.training.yaml_arguments.validate_yaml', validate_args_wrapper)
        pm.register_patch('megatron.training.yaml_arguments._print_args', print_args_wrapper)

        # initialization patches
        from mindspeed.core.megatron_basic.megatron_basic import _compile_dependencies, get_device_wrapper
        pm.register_patch('megatron.training.initialize._compile_dependencies', _compile_dependencies)
        pm.register_patch('megatron.training.dist_signal_handler.get_device', get_device_wrapper)

        from mindspeed.core.megatron_basic.megatron_basic import get_device_arch_version, report_memory, get_model_wrapper
        pm.register_patch('megatron.training.utils.get_device_arch_version', get_device_arch_version)
        pm.register_patch('megatron.training.utils.report_memory', report_memory)
        pm.register_patch('megatron.training.training.get_model', get_model_wrapper)
