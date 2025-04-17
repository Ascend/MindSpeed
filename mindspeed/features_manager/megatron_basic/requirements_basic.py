import sys

import torch
from mindspeed.features_manager.feature import MindSpeedFeature


class RequirementsBasicFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('requirements-basic', optimization_level=0)

    def pre_register_patches(self, patch_manager, args):
        self.te_adaptation(patch_manager, args)
        self.apex_adaptation(patch_manager, args)
        self.torch_adaptation(patch_manager, args)

    def te_adaptation(self, pm, args):
        from mindspeed.core.megatron_basic.requirements_basic import version_wrapper

        pm.register_patch('torch.compile', torch.jit.script)
        # Need replace modules before import megatron
        pm.register_patch('importlib.metadata.version', version_wrapper)
        pm.register_patch('transformer_engine.pytorch.LayerNormLinear', torch.nn.Module, create_dummy=True)
        pm.register_patch('transformer_engine.pytorch.DotProductAttention', torch.nn.Module, create_dummy=True)
        pm.register_patch('transformer_engine.pytorch.Linear', torch.nn.Module, create_dummy=True)
        pm.register_patch('transformer_engine.pytorch.distributed.CudaRNGStatesTracker', torch.nn.Module, create_dummy=True)
        pm.register_patch('transformer_engine.common.recipe.DelayedScaling', torch.nn.Module, create_dummy=True)
        pm.register_patch('flash_attn.flash_attn_interface.flash_attn_unpadded_func', create_dummy=True)

    def apex_adaptation(self, pm, args):
        from mindspeed.core.megatron_basic.requirements_basic import multi_tensor_l2norm, multi_tensor_scale, multi_tensor_applier
        from mindspeed.core.fusions.fused_layer_norm import fused_layer_norm_affine
        from mindspeed.core.optimizer.adamw import FusedTorchAdamW, AdamW
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

        if args.optimizer_selection == "fused_torch_adamw":
            pm.register_patch(
                "apex.optimizers.FusedAdam", FusedTorchAdamW, create_dummy=True
            )
        elif args.optimizer_selection == "fused_adamw":
            pm.register_patch("apex.optimizers.FusedAdam", AdamW, create_dummy=True)
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
