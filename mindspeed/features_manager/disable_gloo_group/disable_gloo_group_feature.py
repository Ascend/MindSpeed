# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from mindspeed.features_manager.feature import MindSpeedFeature


class DisableGlooGroupFeature(MindSpeedFeature):
    def __init__(self):
        super(DisableGlooGroupFeature, self).__init__('disable-gloo-group')

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--disable-gloo-group', action='store_true', default=False,
                           help='Replace the communication method of the DP group in the distributed optimizer from gloo to hccl.')
        group.add_argument('--hccl-slice-size', type=int, default=10 * 1024 * 1024,
                           help='data slice size on each dp rank in distributed optimizer')

    def register_patches(self, patch_manager, args):
        from mindspeed.optimizer.distrib_optimizer import get_parameter_state_dp_zero_hccl, \
            load_parameter_state_from_dp_zero_hccl
        from mindspeed.core.parallel_state import (get_data_parallel_group_gloo_replace,
                                                   get_data_modulo_expert_parallel_group_gloo_replace,
                                                   new_group_wrapper)
        from mindspeed.utils import check_param_hashes_across_dp_replicas_hccl
        if getattr(args, self.feature_name, None):
            patch_manager.register_patch(
                'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.get_parameter_state_dp_zero',
                get_parameter_state_dp_zero_hccl)
            patch_manager.register_patch(
                'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.load_parameter_state_from_dp_zero',
                load_parameter_state_from_dp_zero_hccl)
            patch_manager.register_patch('megatron.core.utils.check_param_hashes_across_dp_replicas',
                                         check_param_hashes_across_dp_replicas_hccl)

            patch_manager.register_patch('megatron.core.parallel_state.get_data_parallel_group_gloo',
                                         get_data_parallel_group_gloo_replace)
            patch_manager.register_patch('megatron.core.parallel_state.get_data_modulo_expert_parallel_group_gloo',
                                         get_data_modulo_expert_parallel_group_gloo_replace)
            patch_manager.register_patch('torch.distributed.new_group', new_group_wrapper)
    
    def validate_args(self, args):
        args.enable_gloo_process_groups = False
        return