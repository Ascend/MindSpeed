from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class ChimeraFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__("schedules-method")
    
    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--schedules-method', type=str,
                           default=None, choices=['dualpipev', "chimera"])
        group.add_argument("--virtual-data-parallel-size", type=int, default=None, help="Number of duplicate pipeline in a stage")
        group.add_argument("--chimera-decouple-bw", action="store_true", default=False, help="Enable decouple bw in the backward when cool down")
    
    def validate_args(self, args):
        if args.schedules_method == "chimera":
            if args.virtual_data_parallel_size > args.pipeline_model_parallel_size:
                raise RuntimeError(
                    "virtual_data_parallel_size should be smaller than or equal to pipeline_model_parallel_size with chimera schedule"
                )
            if args.num_layers_per_virtual_pipeline_stage is not None:
                raise RuntimeError(
                    "The chimera and virtual_pipeline are incompatible."
                )
            if args.pipeline_model_parallel_size <= 1:
                raise RuntimeError(
                    "pipeline_model_parallel_size should be larger than 1 with chimera schedules"
                )
            if args.num_layers % args.pipeline_model_parallel_size != 0:
                raise RuntimeError(
                    "number of layers must be a multiple of pipeline_model_parallel_size"
                )
            if args.virtual_data_parallel_size is None or args.virtual_data_parallel_size < 2:
                raise RuntimeError(
                    f"When enabling chimera, virtual_data_parallel_size must be ≥ 2, but got {args.virtual_data_parallel_size}"
                )
            if (args.virtual_data_parallel_size & (args.virtual_data_parallel_size - 1)) != 0:
                raise RuntimeError(
                    f"virtual_data_parallel_size must be a power of 2, but got {args.virtual_data_parallel_size}"
                )
            num_micro_batch = args.global_batch_size // args.micro_batch_size // args.data_parallel_size
            if num_micro_batch % args.virtual_data_parallel_size != 0:
                raise RuntimeError(
                    f"num_micro_batches ({num_micro_batch}) must be a multiple of virtual_data_parallel_size ({args.virtual_data_parallel_size})"
                )
            print(
                "Warning: The megatron-core MoE with dispatcher type 'alltoall' may have bugs. "
                "Using 'allgather' is recommended. This might be caused by upstream implementation issues."
            )
    
    def register_patches(self, patch_manager, args):
        if args.schedules_method == "chimera":
            from mindspeed.core.pipeline_parallel.chimera import (
                is_pipeline_first_stage_wrapper, 
                is_pipeline_last_stage_wrapper, 
                get_model_wrapper, 
                broadcast_params_wrapper,
                get_embedding_group_wrapper, 
                is_rank_in_embedding_group_wrapper,
                linear_backward_wgrad_detach_wrapper,
                build_pretraining_data_loader_wrapper,
                build_train_valid_test_data_iterators_wrapper,
                make_param_hook_wrapper
            )
            patch_manager.register_patch('megatron.training.training.get_model', get_model_wrapper)
            patch_manager.register_patch('megatron.core.parallel_state.is_pipeline_first_stage', is_pipeline_first_stage_wrapper)
            patch_manager.register_patch('megatron.core.parallel_state.is_pipeline_last_stage', is_pipeline_last_stage_wrapper)
            patch_manager.register_patch("megatron.training.training.build_train_valid_test_data_iterators", build_train_valid_test_data_iterators_wrapper)
            patch_manager.register_patch("megatron.legacy.data.data_samplers.build_pretraining_data_loader", build_pretraining_data_loader_wrapper)
            patch_manager.register_patch('megatron.core.parallel_state.get_embedding_group', get_embedding_group_wrapper)
            patch_manager.register_patch('megatron.core.parallel_state.is_rank_in_embedding_group', is_rank_in_embedding_group_wrapper)
            patch_manager.register_patch('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.broadcast_params', broadcast_params_wrapper)
            if args.chimera_decouple_bw:
                patch_manager.register_patch('megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward',
                                         linear_backward_wgrad_detach_wrapper)
                patch_manager.register_patch('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel._make_param_hook', make_param_hook_wrapper)
            print("Chimera Patch execute successfully...", flush=True)
