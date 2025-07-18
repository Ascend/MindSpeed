# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import sys
from argparse import ArgumentParser

import torch
from mindspeed.features_manager.feature import MindSpeedFeature


class RequirementsBasicFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('requirements-basic', optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--optimizer-selection', type=str, default='fused_adamw',
                           choices=['fused_adamw', 'fused_torch_adamw', 'fused_ema_adamw'],
                           help='Select from the former fused AdamW optimizer and Torch fused AdamW optimizer')
        group.add_argument('--optimization-level', type=int, choices=[0, 1, 2], default=2,
                           help='0: The minimum patch set for megatron to adapt to NPU,'
                                '1: Affinity optimization (fusion operator, etc.), '
                                '2: Advanced acceleration algorithm')

    def pre_register_patches(self, patch_manager, args):
        self.te_adaptation(patch_manager, args)
        self.apex_adaptation(patch_manager, args)
        self.torch_adaptation(patch_manager, args)
        self.optimizer_selection(patch_manager, args)

    def te_adaptation(self, pm, args):
        from mindspeed.core.megatron_basic.requirements_basic import version_wrapper, dummy_compile
        from mindspeed.te.pytorch import MindSpeedTELayernorm
        pm.register_patch('torch.compile', dummy_compile)
        pm.register_patch('torch.jit.script', dummy_compile)
        # Need replace modules before import megatron
        pm.register_patch('importlib.metadata.version', version_wrapper)
        pm.register_patch('transformer_engine.pytorch.LayerNorm', MindSpeedTELayernorm, create_dummy=True)
        pm.register_patch('transformer_engine.pytorch.LayerNormLinear', torch.nn.Module, create_dummy=True)
        pm.register_patch('transformer_engine.pytorch.DotProductAttention', torch.nn.Module, create_dummy=True)
        pm.register_patch('transformer_engine.pytorch.Linear', torch.nn.Module, create_dummy=True)
        pm.register_patch('transformer_engine.pytorch.GroupedLinear', torch.nn.Module, create_dummy=True)
        pm.register_patch('transformer_engine.pytorch.distributed.CudaRNGStatesTracker', torch.nn.Module, create_dummy=True)
        pm.register_patch('transformer_engine.common.recipe.DelayedScaling', torch.nn.Module, create_dummy=True)
        pm.register_patch('flash_attn.flash_attn_interface.flash_attn_unpadded_func', create_dummy=True)

    def apex_adaptation(self, pm, args):
        from mindspeed.core.megatron_basic.requirements_basic import multi_tensor_l2norm, multi_tensor_scale, multi_tensor_applier
        from mindspeed.core.fusions.fused_layer_norm import fused_layer_norm_affine
        from mindspeed.ops.npu_matmul_add import npu_matmul_add_fp32, npu_matmul_add_fp16
        from mindspeed.core.fusions.fused_layer_norm import FusedLayerNormAffineFunction, FastLayerNormFN

        pm.register_patch('amp_C.multi_tensor_l2norm', multi_tensor_l2norm, create_dummy=True)
        pm.register_patch('amp_C.multi_tensor_scale', multi_tensor_scale, create_dummy=True)
        pm.register_patch('apex.multi_tensor_apply.multi_tensor_applier', multi_tensor_applier, create_dummy=True)
        pm.register_patch('apex.normalization.fused_layer_norm.fused_layer_norm_affine', fused_layer_norm_affine, create_dummy=True)
        pm.register_patch('fused_layer_norm_cuda', create_dummy=True)
        pm.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32', npu_matmul_add_fp32, create_dummy=True)
        pm.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16', npu_matmul_add_fp16, create_dummy=True)
        pm.register_patch('apex.contrib.layer_norm.layer_norm.FastLayerNormFN', FastLayerNormFN, create_dummy=True)
        pm.register_patch('apex.normalization.fused_layer_norm.FusedLayerNormAffineFunction',
                          FusedLayerNormAffineFunction, create_dummy=True)
        
    def optimizer_selection(self, pm, args):
        from mindspeed.core.optimizer.adamw import FusedTorchAdamW, AdamW
        if args.optimizer_selection == 'fused_torch_adamw':
            pm.register_patch('apex.optimizers.FusedAdam', FusedTorchAdamW, create_dummy=True)
        elif args.optimizer_selection == 'fused_adamw':
            pm.register_patch('apex.optimizers.FusedAdam', AdamW, create_dummy=True)
        pm.register_patch('apex.optimizers.FusedSGD', torch.optim.SGD, create_dummy=True)

    def torch_adaptation(self, pm, args):
        from torch.distributed import all_gather_into_tensor, reduce_scatter_tensor
        from mindspeed.core.megatron_basic.requirements_basic import type_wrapper, ensure_contiguous_wrapper, lcm, \
            dummy_function, torch_all_reduce_double_dtype_bypass_wrapper

        pm.register_patch('torch.nn.parameter.Parameter.type', type_wrapper)
        pm.register_patch('torch.Tensor.type', type_wrapper)
        pm.register_patch('torch.Tensor.view', ensure_contiguous_wrapper)
        pm.register_patch('torch.distributed._all_gather_base', all_gather_into_tensor)
        pm.register_patch('torch.distributed._reduce_scatter_base', reduce_scatter_tensor)
        pm.register_patch('torch.distributed.all_reduce', torch_all_reduce_double_dtype_bypass_wrapper)
        pm.register_patch('torch._C._jit_set_nvfuser_enabled', dummy_function)
        # lmc is supported python >=3.9
        if sys.version_info < (3, 9):
            pm.register_patch('math.lcm', lcm, create_dummy=True)
