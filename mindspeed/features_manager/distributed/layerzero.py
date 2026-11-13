# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""Define LayerZero feature.

Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""

from argparse import ArgumentParser, Namespace
from logging import getLogger

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager

LOG = getLogger(__name__)


class LayerZeroFeature(MindSpeedFeature):
    """Layer Zero feature."""

    def __init__(
        self,
        feature_name: str = "layerzero",
        optimization_level: int = 2,
    ):
        super().__init__(feature_name, optimization_level)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            '--layerzero',
            action='store_true',
            default=False,
            help='Use this flag to enable layerzero, including the segmentation of the parameters, gradients, and '
                 'optimizers of the row-parallel and column-parallel models, as well as the overlap optimization of '
                 'the gradient reduce sactter and weight all gather.')
        group.add_argument('--layerzero-config', type=str,
                           help='Use this yaml file to config layerzero behaviours')

    def register_patches(
        self,
        patch_manager: MindSpeedPatchesManager,
        args: Namespace,
    ):
        # pylint: disable=import-outside-toplevel
        from mindspeed.core.distributed.layerzero import (layerzero_setup_model_and_optimizer,
                                                          layerzero_initialize_model_parallel_wrapper,
                                                          mga_finalize_model_grads,
                                                          save_checkpoint,
                                                          )
        patch_manager.register_patch('megatron.training.training.setup_model_and_optimizer',
                                     layerzero_setup_model_and_optimizer)
        patch_manager.register_patch('megatron.core.parallel_state.initialize_model_parallel',
                                     layerzero_initialize_model_parallel_wrapper)
        patch_manager.register_patch('megatron.core.distributed.finalize_model_grads',
                                     mga_finalize_model_grads)
        patch_manager.register_patch('megatron.training.checkpointing.save_checkpoint',
                                     save_checkpoint)
