# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from mindspeed.features_manager.feature import MindSpeedFeature


class CustomFSDPFeature(MindSpeedFeature):
    def __init__(self):
        super(CustomFSDPFeature, self).__init__('use-megatron-fsdp')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.distributed.custom_fsdp.param_and_grad_buffer import gradient_reduce_preprocessing, _bucket_group_gradient_reduce
        from mindspeed.moe.router import gating

        patch_manager.register_patch('megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer.gradient_reduce_preprocessing', 
                                        gradient_reduce_preprocessing)
        patch_manager.register_patch('megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer.GradReducePipeline._bucket_group_gradient_reduce', 
                                        _bucket_group_gradient_reduce)
        patch_manager.register_patch('megatron.core.transformer.moe.router.Router.gating', gating)
