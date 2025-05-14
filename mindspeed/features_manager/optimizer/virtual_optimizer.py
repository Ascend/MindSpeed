# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser

import torch

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
            if not hasattr(torch_npu, "empty_with_swapped_memory"):
                raise AssertionError("`--virtual-optimizer` is invalid, please update the latest PTA version.")
        self.incompatible_check(args, "fused_ema_adamw")

    
    def register_patches(self, patch_manager, args):
        from mindspeed.core.optimizer.virtual_optimizer.adaptor import virtual_optimizer_step, replace_swap_tensor_wrapper
        if getattr(args, self.feature_name, None):
            patch_manager.register_patch('mindspeed.optimizer.adamw.AdamW.step', virtual_optimizer_step)
            patch_manager.register_patch(
                'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.load_parameter_state_from_dp_zero',
                replace_swap_tensor_wrapper)
            patch_manager.register_patch(
                'megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params.load_state_dict', replace_swap_tensor_wrapper)
            # adapt to save/load ckpt
            torch.Tensor.copy_ = swap_tensor_copy_wrapper(torch.Tensor.copy_)
            torch.Tensor.cpu = swap_tensor_func_wrapper(torch.Tensor.cpu, "cpu")
            torch.Tensor.clone = swap_tensor_func_wrapper(torch.Tensor.clone, "clone")
            torch.Tensor.npu = swap_tensor_func_wrapper(torch.Tensor.npu, "npu")
            torch.Tensor.detach = swap_tensor_func_wrapper(torch.Tensor.detach, "detach")


def is_swap_tensor(tensor: torch.Tensor):
    return hasattr(tensor, "swap_tensor") and tensor.swap_tensor
 
 
def swap_tensor_copy_wrapper(func):
    def wrapped(*args, **kwargs):
        dst, src = args[0], args[1]
        dst_swap, src_swap = is_swap_tensor(dst), is_swap_tensor(src)
        if dst_swap or src_swap:
            if dst.device == src.device:
                dst.fill_(1).mul_(src)
            elif dst_swap:
                src_npu = src.to(dst.device)
                dst.fill_(1).mul_(src_npu)
            elif src_swap:
                src_npu = torch.ones_like(src).mul(src)
                dst.copy_(src_npu)
            else:
                raise TypeError
        else:
            func(*args, **kwargs)
    return wrapped

 
def swap_tensor_func_wrapper(org_func, func_type):
    def wrapped(*args, **kwargs):
        if is_swap_tensor(args[0]):
            if func_type == "detach":
                detach = org_func(*args, **kwargs)
                setattr(detach, "swap_tensor", True)
                setattr(detach.data, "swap_tensor", True)
                return detach
            src = torch.empty_like(args[0])
            src.copy_(args[0])
            if func_type == "cpu":
                return src.cpu()
            elif func_type == "npu" or func_type == "clone":
                return src
            else:
                raise ValueError(f"func_type {func_type} not supported")
        else:
            return org_func(*args, **kwargs)
    return wrapped

