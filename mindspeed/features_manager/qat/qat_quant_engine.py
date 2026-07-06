# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
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
            choices=['w4a16-mxfp4', 'w4a16-mxfp4-moe-only', 'w8a16-mxfp8', 'w8a16-mxfp8-moe-only', 'w4a4-mxfp4'],
            help='Set the QAT quantization method',
        )

    def register_patches(self, patch_manager: MindSpeedPatchesManager, args):
        scheme = getattr(args, 'qat_scheme', None)
        if scheme in ["w4a16-mxfp4", "w8a16-mxfp8", 'w4a4-mxfp4']:
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
                elif scheme == "w4a4-mxfp4":
                    from mindspeed.core.qat.layers import (
                        linear_with_grad_accumulation_and_async_w4a4_forward as forward_func,
                    )
                    from mindspeed.core.qat.layers import (
                        linear_with_grad_accumulation_and_async_w4a4_backward as backward_func,
                    )
                else:
                    return
                patch_manager.register_patch(
                    'megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.forward',
                    forward_func,
                )
                patch_manager.register_patch(
                    'megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward',
                    backward_func,
                )
