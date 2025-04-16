# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class VirtualOptimizerFeature(MindSpeedFeature):
    
    def __init__(self):
        super().__init__("virtual-optimizer", 2)
    
    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        
        def parse_list_for_virtual_optimizer(value):
            if value == 'all':
                return 65.0 # Maximum NPU memory
            try:
                return float(value)
            except ValueError as e:
                print(f"--virtual-optimizer has invalid value: {value}. Expected 'all' or a float/int numer.")
                raise e
        group.add_argument(
            '--virtual-optimizer', 
            type=parse_list_for_virtual_optimizer, 
            nargs='+', 
            help="User vritual memory to swap Optimizer. Pass a list of 'all' or values, e.g. 'all' or '1', '2'")
    
    def validate_args(self, args):
        if args.virtual_optimizer is not None:
            import torch_npu
            if not hasattr(torch_npu, "empty_with_swap_memory"):
                raise AssertionError("`--virtual-optimizer` is invalid, please update the latest PTA version.")
        self.incompatible_check(args, "fused_ema_adamw")

    
    def register_patches(self, patch_manager, args):
        from mindspeed.core.optimizer.virtual_optimizer.adaptor import virtual_optimizer_step
        if getattr(args, self.feature_name, None):
            patch_manager.register_patch('mindspeed.optimizer.adamw.AdamW.step', virtual_optimizer_step)
