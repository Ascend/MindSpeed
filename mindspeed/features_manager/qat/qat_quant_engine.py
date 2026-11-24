# Copyright 2025 Huawei Technologies Co., Ltd.
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
import warnings


from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class QATQuantEngineFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('qat-quant-engine', optimization_level=2)

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            '--qat-scheme',
            type=str,
            default=None,
            choices=['w4a16-mxfp4', 'w4a16-mxfp4-moe-only', 'w8a16-mxfp8', 'w8a16-mxfp8-moe-only'],
            help='Set the QAT quantization method',
        )

    def register_patches(self, pm: MindSpeedPatchesManager, args):
        scheme = getattr(args, 'qat_scheme', None)
        if scheme in ["w4a16-mxfp4", "w8a16-mxfp8"]:
            use_optimized_linear = (
                getattr(args, "gradient_accumulation_fusion", False)
                or getattr(args, "async_tensor_model_parallel_allreduce", False)
                or getattr(args, "sequence_parallel", False)
            )
            if not use_optimized_linear:
                warnings.warn(
                    f"{scheme} quantization requires at least one of the following optimizations "
                    f"to be enabled to use the optimized linear layer: "
                    f"--gradient-accumulation-fusion, --async-tensor-model-parallel-allreduce, "
                    f"--sequence-parallel. "
                )
            else:
                if scheme == "w4a16-mxfp4":
                    from mindspeed.core.qat.layers import (
                        linear_with_grad_accumulation_and_async_w4a16_forward as forward_func,
                    )
                    from mindspeed.core.qat.layers import (
                        linear_with_grad_accumulation_and_async_w4a16_backward as backward_func,
                    )
                elif scheme == "w8a16-mxfp8":
                    from mindspeed.core.qat.layers import (
                        linear_with_grad_accumulation_and_async_w8a16_forward as forward_func,
                    )
                    from mindspeed.core.qat.layers import (
                        linear_with_grad_accumulation_and_async_w8a16_backward as backward_func,
                    )
                else:
                    return
                pm.register_patch(
                    'megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.forward',
                    forward_func,
                )
                pm.register_patch(
                    'megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward',
                    backward_func,
                )
