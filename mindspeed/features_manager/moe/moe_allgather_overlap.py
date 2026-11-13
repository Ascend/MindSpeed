# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser
import torch
from mindspeed.features_manager.feature import MindSpeedFeature


class MoEAllGatherOverLapFeature(MindSpeedFeature):
    '''
    MoE Layer AllGather OverLap spec.
    This spec supports "allgather" dispatcher.
    '''
    def __init__(self):
        super().__init__('moe-allgather-overlap-comm', 2)
 
    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--moe-allgather-overlap-comm', action='store_true', default=False,
                        help='Use async communication&swap to overlap compute in allgather.')

    def validate_args(self, args):
        self.incompatible_check(args, 'use_ascend_mc2')
        if args.moe_allgather_overlap_comm and not args.moe_token_dispatcher_type == 'allgather':
            raise AssertionError('`--moe-allgather-overlap-comm` only support with `--moe-token-dispatcher-type allgather`.')
        if args.moe_allgather_overlap_comm:
            if not getattr(args, 'moe_permutation_async_comm'):
                raise AssertionError('`--moe-alltoall-overlap-comm` and `--moe-allgather-overlap-comm` only support with `--moe-permutation-async-comm`.')
            if not args.moe_grouped_gemm:
                raise AssertionError('`--moe-alltoall-overlap-comm` and `--moe-allgather-overlap-comm` only support with `--moe-grouped-gemm`.')

            #Convert Megatron Shared_experts to MindSpeed version. This convert operation only for some judge.
            if args.n_shared_experts is None and args.moe_shared_expert_intermediate_size is not None:
                args.n_shared_experts = args.moe_shared_expert_intermediate_size // (
                    args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else args.ffn_hidden_size)

    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.moe.moe_feature.adaptor import MindSpeedAllGatherOverlapMoeLayerAdaptor
        from mindspeed.core.transformer.moe.moe_feature.overlap.moe_common import mlp_init, parallel_transformer_layer_init_wrapper, core_mlp_forward_wrapper
        patch_manager.register_patch('megatron.core.transformer.mlp.MLP.forward',
                    core_mlp_forward_wrapper)
        if getattr(args, 'moe_token_dispatcher_type', None) == "allgather":
            if args.moe_allgather_overlap_comm:
                patch_manager.register_patch(
                    'megatron.core.transformer.moe.moe_layer.MoELayer',
                    MindSpeedAllGatherOverlapMoeLayerAdaptor)
                patch_manager.register_patch(
                    'megatron.core.transformer.mlp.MLP.__init__',
                    mlp_init)
                patch_manager.register_patch(
                    'megatron.core.transformer.transformer_layer.TransformerLayer.__init__',
                    parallel_transformer_layer_init_wrapper)
