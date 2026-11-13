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

import torch

from mindspeed.args_utils import get_full_args as get_args
from .current_scaling_recipe import CurrentScalingRecipe, Float8CurrentScaling, TensorwiseMatMul
from .delayed_scaling_recipe import DelayedScalingRecipe, TEDelayedScaling
from .float8_block_scaling_recipe import Float8BlockRecipe, Float8BlockScaling, Float8BlockMatMul
from .mxfp8_scaling_recipe import MXFP8ScalingRecipe, MXFP8BlockScaling, MXFP8MatMul
from .mxfp8_32x32_scaling_recipe import MXFP832x32ScalingRecipe, MXFP832x32BlockScaling, MXFP832x32MatMul
from ..constants import Fp8Recipe

SCALING_TYPE_MAP = {
    Fp8Recipe.delayed: DelayedScalingRecipe,
    Fp8Recipe.tensorwise: CurrentScalingRecipe,
    Fp8Recipe.mxfp8: MXFP8ScalingRecipe,
    Fp8Recipe.mxfp8_32x32: MXFP832x32ScalingRecipe,
    Fp8Recipe.blockwise: Float8BlockRecipe,
}

SCALING_CONFIG_MAP = {
    Fp8Recipe.delayed: TEDelayedScaling,
    Fp8Recipe.tensorwise: Float8CurrentScaling,
    Fp8Recipe.mxfp8: MXFP8BlockScaling,
    Fp8Recipe.mxfp8_32x32: MXFP832x32BlockScaling,
    Fp8Recipe.blockwise: Float8BlockScaling,
}

MATMUL_MAP = {
    Fp8Recipe.mxfp8: MXFP8MatMul,
    Fp8Recipe.mxfp8_32x32: MXFP832x32MatMul,
    Fp8Recipe.tensorwise: TensorwiseMatMul,
    Fp8Recipe.delayed: TensorwiseMatMul,
    Fp8Recipe.blockwise: Float8BlockMatMul,
}


def matmul_fp8(inputs: torch.Tensor, weight: torch.Tensor):
    if get_args().fp8_recipe not in MATMUL_MAP:
        return torch.matmul(inputs, weight.t())
    if weight.requires_grad:
        # overlap 需要手动重计算, 需要额外判断重计算阶段
        from mindspeed.te.pytorch.fp8 import is_fp8_activation_recompute_enabled, in_fp8_activation_recompute_phase

        need_grad = not is_fp8_activation_recompute_enabled() or in_fp8_activation_recompute_phase()
    else:
        need_grad = False

    return MATMUL_MAP[get_args().fp8_recipe].apply(inputs, weight, need_grad)
