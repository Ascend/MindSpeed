from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class DistTrainFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('dist-train', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--dist-train', action='store_true', help='Enable dist-train feature.')

    def validate_args(self, args):
        if args.dist_train:
            if not hasattr(args, 'mm_model'):
                raise ValueError('DistTrain must work with MindSpeed-MM')
            from mindspeed.core.multi_modal.dist_train.dist_train_config import validate_configs_world_size, \
                get_dist_model_config, merge_dist_train_args
            merge_dist_train_args(args.mm_model)
            validate_configs_world_size(args)
            cfg = get_dist_model_config(rank=args.rank)
            args.world_size = cfg.world_size
            args.tensor_model_parallel_size = cfg.tensor_model_parallel_size
            args.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
            args.context_parallel_size = cfg.context_parallel_size
        seq_parallel_enabled = args.sequence_parallel
        if args.tensor_model_parallel_size > 1 and seq_parallel_enabled:
            args.sequence_parallel = True
        from mindspeed.core.multi_modal.dist_train.dist_train_config import get_all_config
        if any(cfg.main_dp for cfg in get_all_config().values()):
            from mindspeed.core.multi_modal.dist_train.utils import get_global_data_parallel_size
            args.data_parallel_size = get_global_data_parallel_size()

    def register_patches(self, patch_manager, args):
        if args.dist_train:
            from mindspeed.core.multi_modal import dist_train
            # pipeline parallel adaption
            patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.get_forward_backward_func',
                                         dist_train.dist_schedules.get_forward_backward_func_wrapper)
            patch_manager.register_patch('megatron.core.pipeline_parallel.p2p_communication._p2p_ops',
                                         dist_train.dist_schedules.p2p_ops_wrapper)
            # parallel state adaption
            patch_manager.register_patch('megatron.training.initialize._initialize_distributed',
                                         dist_train.dist_schedules.initialize_distributed_wrapper)
            patch_manager.register_patch('megatron.core.mpu.initialize_model_parallel',
                                         dist_train.dist_parallel_state.initialize_model_parallel)
            patch_manager.register_patch('megatron.core.mpu.is_pipeline_last_stage',
                                         dist_train.dist_parallel_state.get_is_pipeline_last_stage_wrapper)
            patch_manager.register_patch('megatron.core.mpu.is_pipeline_first_stage',
                                         dist_train.dist_parallel_state.get_is_pipeline_first_stage_wrapper)
            patch_manager.register_patch('megatron.core.mpu.get_tensor_model_parallel_src_rank',
                                         dist_train.dist_parallel_state.get_tensor_model_parallel_src_rank_wrapper)
            patch_manager.register_patch('megatron.core.mpu.is_initialized',
                                         dist_train.dist_parallel_state.is_initialized)
            patch_manager.register_patch('megatron.core.mpu.model_parallel_is_initialized',
                                         dist_train.dist_parallel_state.model_parallel_is_initialized)
            patch_manager.register_patch('megatron.core.mpu.get_model_parallel_group',
                                         dist_train.dist_parallel_state.get_model_parallel_group)
            patch_manager.register_patch('megatron.core.mpu.get_tensor_model_parallel_group',
                                         dist_train.dist_parallel_state.get_tensor_model_parallel_group)
            patch_manager.register_patch('megatron.core.mpu.get_pipeline_model_parallel_group',
                                         dist_train.dist_parallel_state.get_pipeline_model_parallel_group)
            patch_manager.register_patch('megatron.core.mpu.get_data_parallel_group',
                                         dist_train.dist_parallel_state.get_data_parallel_group)
            patch_manager.register_patch('megatron.core.mpu.get_data_parallel_group_gloo',
                                         dist_train.dist_parallel_state.get_data_parallel_group_gloo)
            patch_manager.register_patch('megatron.core.mpu.get_inter_partial_data_parallel_group',
                                         dist_train.dist_parallel_state.get_inter_partial_data_parallel_group)
            patch_manager.register_patch('megatron.core.mpu.get_context_parallel_group',
                                         dist_train.dist_parallel_state.get_context_parallel_group)
            patch_manager.register_patch('megatron.core.mpu.get_context_parallel_global_ranks',
                                         dist_train.dist_parallel_state.get_context_parallel_global_ranks)
            patch_manager.register_patch('megatron.core.mpu.get_hierarchical_context_parallel_groups',
                                         dist_train.dist_parallel_state.get_hierarchical_context_parallel_groups)
            patch_manager.register_patch('megatron.core.mpu.get_embedding_group',
                                         dist_train.dist_parallel_state.get_embedding_group)
            patch_manager.register_patch('megatron.core.mpu.get_position_embedding_group',
                                         dist_train.dist_parallel_state.get_position_embedding_group)
            patch_manager.register_patch('megatron.core.mpu.get_amax_reduction_group',
                                         dist_train.dist_parallel_state.get_amax_reduction_group)
            patch_manager.register_patch('megatron.core.mpu.get_tensor_and_data_parallel_group',
                                         dist_train.dist_parallel_state.get_tensor_and_data_parallel_group)
            patch_manager.register_patch('megatron.core.mpu.get_tensor_and_context_parallel_group',
                                         dist_train.dist_parallel_state.get_tensor_and_context_parallel_group)
            patch_manager.register_patch('megatron.core.mpu.get_tensor_model_parallel_world_size',
                                         dist_train.dist_parallel_state.get_tensor_model_parallel_world_size)
            patch_manager.register_patch('megatron.core.mpu.get_pipeline_model_parallel_world_size',
                                         dist_train.dist_parallel_state.get_pipeline_model_parallel_world_size)
            patch_manager.register_patch('megatron.core.mpu.get_tensor_model_parallel_rank',
                                         dist_train.dist_parallel_state.get_tensor_model_parallel_rank)
            patch_manager.register_patch('megatron.core.mpu.get_pipeline_model_parallel_rank',
                                         dist_train.dist_parallel_state.get_pipeline_model_parallel_rank)
            patch_manager.register_patch('megatron.core.mpu.get_pipeline_model_parallel_split_rank',
                                         dist_train.dist_parallel_state.get_pipeline_model_parallel_split_rank)
            patch_manager.register_patch('megatron.core.mpu.is_rank_in_embedding_group',
                                         dist_train.dist_parallel_state.is_rank_in_embedding_group)
            patch_manager.register_patch('megatron.core.mpu.is_rank_in_position_embedding_group',
                                         dist_train.dist_parallel_state.is_rank_in_position_embedding_group)
            patch_manager.register_patch('megatron.core.mpu.get_virtual_pipeline_model_parallel_rank',
                                         dist_train.dist_parallel_state.get_virtual_pipeline_model_parallel_rank)
            patch_manager.register_patch('megatron.core.mpu.get_virtual_pipeline_model_parallel_world_size',
                                         dist_train.dist_parallel_state.get_virtual_pipeline_model_parallel_world_size)
            patch_manager.register_patch('megatron.core.mpu.get_model_parallel_src_rank',
                                         dist_train.dist_parallel_state.get_model_parallel_src_rank)
            patch_manager.register_patch('megatron.core.mpu.get_data_parallel_src_rank',
                                         dist_train.dist_parallel_state.get_data_parallel_src_rank)
            patch_manager.register_patch('megatron.core.mpu.get_pipeline_model_parallel_first_rank',
                                         dist_train.dist_parallel_state.get_pipeline_model_parallel_first_rank)
            patch_manager.register_patch('megatron.core.mpu.get_pipeline_model_parallel_last_rank',
                                         dist_train.dist_parallel_state.get_pipeline_model_parallel_last_rank)
            patch_manager.register_patch('megatron.core.mpu.get_pipeline_model_parallel_next_rank',
                                         dist_train.dist_parallel_state.get_pipeline_model_parallel_next_rank)
            patch_manager.register_patch('megatron.core.mpu.get_pipeline_model_parallel_prev_rank',
                                         dist_train.dist_parallel_state.get_pipeline_model_parallel_prev_rank)
            patch_manager.register_patch('megatron.core.mpu.get_data_parallel_world_size',
                                         dist_train.dist_parallel_state.get_data_parallel_world_size)
            patch_manager.register_patch('megatron.core.mpu.get_data_parallel_rank',
                                         dist_train.dist_parallel_state.get_data_parallel_rank)
            patch_manager.register_patch('megatron.core.mpu.get_context_parallel_world_size',
                                         dist_train.dist_parallel_state.get_context_parallel_world_size)
            patch_manager.register_patch('megatron.core.mpu.get_context_parallel_rank',
                                         dist_train.dist_parallel_state.get_context_parallel_rank)
            patch_manager.register_patch('megatron.core.mpu.get_tensor_and_context_parallel_world_size',
                                         dist_train.dist_parallel_state.get_tensor_and_context_parallel_world_size)
            patch_manager.register_patch('megatron.core.mpu.get_tensor_and_context_parallel_rank',
                                         dist_train.dist_parallel_state.get_tensor_and_context_parallel_rank)
            patch_manager.register_patch('megatron.core.mpu.get_expert_model_parallel_group',
                                         dist_train.dist_parallel_state.get_expert_model_parallel_group)
            patch_manager.register_patch('megatron.core.mpu.get_expert_model_parallel_world_size',
                                         dist_train.dist_parallel_state.get_expert_model_parallel_world_size)
            patch_manager.register_patch('megatron.core.mpu.get_expert_model_parallel_rank',
                                         dist_train.dist_parallel_state.get_expert_model_parallel_rank)
            patch_manager.register_patch('megatron.core.mpu.get_expert_tensor_parallel_group',
                                         dist_train.dist_parallel_state.get_expert_tensor_parallel_group)
            patch_manager.register_patch('megatron.core.mpu.get_expert_tensor_parallel_world_size',
                                         dist_train.dist_parallel_state.get_expert_tensor_parallel_world_size)
            patch_manager.register_patch('megatron.core.mpu.get_expert_tensor_parallel_rank',
                                         dist_train.dist_parallel_state.get_expert_tensor_parallel_rank)
            patch_manager.register_patch('megatron.core.mpu.get_expert_tensor_and_model_parallel_group',
                                         dist_train.dist_parallel_state.get_expert_tensor_and_model_parallel_group)
            patch_manager.register_patch('megatron.core.mpu.get_expert_tensor_and_model_parallel_world_size',
                                         dist_train.dist_parallel_state.get_expert_tensor_and_model_parallel_world_size)
            patch_manager.register_patch('megatron.core.mpu.get_expert_tensor_and_model_parallel_rank',
                                         dist_train.dist_parallel_state.get_expert_tensor_and_model_parallel_rank)
            patch_manager.register_patch('megatron.core.mpu.get_expert_tensor_model_pipeline_parallel_group',
                                         dist_train.dist_parallel_state.get_expert_tensor_model_pipeline_parallel_group)
            patch_manager.register_patch('megatron.core.mpu.get_expert_data_parallel_group',
                                         dist_train.dist_parallel_state.get_expert_data_parallel_group)
            patch_manager.register_patch('megatron.core.mpu.get_data_modulo_expert_parallel_group',
                                         dist_train.dist_parallel_state.get_data_modulo_expert_parallel_group)
            patch_manager.register_patch('megatron.core.mpu.get_expert_data_parallel_group_gloo',
                                         dist_train.dist_parallel_state.get_expert_data_parallel_group_gloo)
            patch_manager.register_patch('megatron.core.mpu.get_expert_data_parallel_rank',
                                         dist_train.dist_parallel_state.get_expert_data_parallel_rank)
            patch_manager.register_patch('megatron.core.mpu.get_global_memory_buffer',
                                         dist_train.dist_parallel_state.get_global_memory_buffer)
            patch_manager.register_patch('megatron.core.mpu.get_moe_layer_wise_logging_tracker',
                                         dist_train.dist_parallel_state.get_moe_layer_wise_logging_tracker)
            # checkpoint
            patch_manager.register_patch('megatron.training.checkpointing.get_checkpoint_name',
                                         dist_train.dist_schedules.get_checkpoint_name_wrapper)
