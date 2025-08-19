# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from mindspeed.features_manager.feature import MindSpeedFeature


class CustomFSDPFeature(MindSpeedFeature):
    def __init__(self):
        super(CustomFSDPFeature, self).__init__('use-custom-fsdp')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.distributed.custom_fsdp.param_and_grad_buffer import gradient_reduce_preprocessing, mark_bucket_ready
        from mindspeed.moe.router import gating

        patch_manager.register_patch('megatron.core.distributed.custom_fsdp.param_and_grad_buffer.gradient_reduce_preprocessing', 
                                        gradient_reduce_preprocessing)
        patch_manager.register_patch('megatron.core.distributed.custom_fsdp.param_and_grad_buffer.GradReducePipeline.mark_bucket_ready', 
                                        mark_bucket_ready)
        patch_manager.register_patch('megatron.core.transformer.moe.router.Router.gating', gating)
