# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class TransformerEngineBasicFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('transformer-engine-basic', optimization_level=0)

    def pre_register_patches(self, pm, args):
        pm.register_patch('transformer_engine.pytorch.tensor.QuantizedTensor', torch.nn.Module, create_dummy=True)

    def register_patches(self, pm: MindSpeedPatchesManager, args):
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
        from megatron.core.transformer.dot_product_attention import DotProductAttention
        from mindspeed.te.pytorch.module.layernorm_column_parallel_linear import MindSpeedTELayerNormColumnParallelLinear
        from mindspeed.te.pytorch.module.grouped_linear import MindSpeedTEGroupedLinear, MindSpeedTEColumnParallelGroupedLinear, MindSpeedTERowParallelGroupedLinear

        pm.register_patch('megatron.core.extensions.transformer_engine.TEColumnParallelLinear', ColumnParallelLinear)
        pm.register_patch('megatron.core.extensions.transformer_engine.TERowParallelLinear', RowParallelLinear)
        # cp == 1
        if int(getattr(args, 'context_parallel_size', 1)) == 1:
            pm.register_patch('megatron.core.extensions.transformer_engine.TEDotProductAttention', DotProductAttention)

        pm.register_patch('megatron.core.extensions.transformer_engine.TELayerNormColumnParallelLinear', MindSpeedTELayerNormColumnParallelLinear)
        pm.register_patch('megatron.core.extensions.transformer_engine.TEGroupedLinear', MindSpeedTEGroupedLinear)
        pm.register_patch('megatron.core.extensions.transformer_engine.TEColumnParallelGroupedLinear', MindSpeedTEColumnParallelGroupedLinear)
        pm.register_patch('megatron.core.extensions.transformer_engine.TERowParallelGroupedLinear', MindSpeedTERowParallelGroupedLinear)
