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

from typing import Optional

import torch
import torch_npu
from torch.library import impl
from mindspeed.op_builder.builder import AS_LIBRARY
from mindspeed.ops import gmm
from mindspeed.ops._gmm_graph import (
    Tensor,
    TensorSpec,
    DataType,
    GroupedMatmul,
    fill_empty_tensor,
    register_fx_node_ge_converter,
)

__all__ = ["npu_quant_gmm", "npu_quant_gmm_v2"]


AS_LIBRARY.define(
    'npu_quant_gmm(Tensor x, Tensor weight, Tensor scale, *, Tensor? offset=None, Tensor? '
    'per_token_scale=None, Tensor? bias=None, Tensor? group_list=None, int? group_list_type=0, '
    'ScalarType? output_dtype=None, int? act_type=0) -> Tensor'
)


@impl(AS_LIBRARY, "npu_quant_gmm", "Meta")
def npu_quant_gmm_forward(
    x,
    weight,
    scale,
    *,
    offset=None,
    per_token_scale=None,
    bias=None,
    group_list=None,
    group_list_type=0,
    output_dtype=None,
    act_type=0,
):
    BM = x.shape[0]
    N = weight.shape[-1]
    output_dtype = output_dtype or torch.bfloat16
    return x.new_empty((BM, N), dtype=output_dtype)


@register_fx_node_ge_converter(torch.ops.mindspeed.npu_quant_gmm.default)
def conveter_npu_quant_gmm(
    x: Tensor,
    weight: Tensor,
    scale: Tensor,
    *,
    offset: Optional[Tensor] = None,
    per_token_scale: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    group_list: Optional[Tensor] = None,
    group_list_type: Optional[int] = 0,
    output_dtype: Optional[DataType] = None,
    act_type: Optional[int] = 0,
    meta_outputs: TensorSpec = None,
):
    bias = bias or fill_empty_tensor(DataType.DT_INT32)
    offset = offset or fill_empty_tensor(DataType.DT_FLOAT)
    antiquant_scale = fill_empty_tensor(DataType.DT_FLOAT16)
    antiquant_offset = fill_empty_tensor(DataType.DT_FLOAT16)

    y_dtype = 0
    if output_dtype == torch.float16:
        y_dtype = 0
    elif output_dtype is None or output_dtype == torch.bfloat16:
        y_dtype = 1
    elif output_dtype == torch.int8:
        raise ValueError("output_dtype not support int8 yet for graph mode")
    else:
        raise ValueError(
            f"output_dtype should be int8, float16 or bfloat16, otherwise it should be None, but got {output_dtype}"
        )

    return GroupedMatmul(
        [x],
        [weight],
        [bias],
        [scale],
        [offset],
        [antiquant_scale],
        [antiquant_offset],
        group_list,
        per_token_scale,
        split_item=3,
        group_type=0,
        group_list_type=group_list_type,
        dtype=y_dtype,
        act_type=act_type,
    )[0]


@impl(AS_LIBRARY, "npu_quant_gmm", "PrivateUse1")
def _npu_quant_gmm(
    x,
    weight,
    scale,
    *,
    offset=None,
    per_token_scale=None,
    bias=None,
    group_list=None,
    group_list_type=0,
    output_dtype=None,
    act_type=0,
):
    output_dtype = torch.bfloat16 if output_dtype is None else output_dtype
    if output_dtype not in (torch.bfloat16, torch.float16, torch.int8):
        raise ValueError(f"output_dtype should be int8, float16, bfloat16 or None, but got {output_dtype}")
    return torch_npu.npu_grouped_matmul.default(
        [x],
        [weight],
        scale=[] if scale is None else [scale],
        offset=[] if offset is None else [offset],
        per_token_scale=[] if per_token_scale is None else [per_token_scale],
        bias=[] if bias is None else [bias],
        group_list=group_list,
        split_item=3,
        group_type=0,
        group_list_type=group_list_type,
        output_dtype=output_dtype,
        act_type=act_type,
    )[0]


def _npu_quant_gmm_common(
    x,
    weight,
    scale,
    *,
    offset=None,
    per_token_scale=None,
    bias=None,
    group_list=None,
    group_list_type=0,
    output_dtype=None,
    act_type=0,
):
    if x.dtype != torch.int8 or weight.dtype != torch.int8:
        raise ValueError(f"Quant gmm only accept quant case, but got x[{x.dtype}] weight[{weight.dtype}]")
    gmm.npu_gmm_param_verification(
        x, weight, bias=bias, group_list=group_list, group_type=0, group_list_type=group_list_type
    )
    gmm.check_optional_tensor(scale, x.device, "scale")
    gmm.check_optional_tensor(offset, x.device, "offset")
    gmm.check_optional_tensor(per_token_scale, x.device, "per_token_scale")
    op = torch.ops.mindspeed.npu_quant_gmm if torch.compiler.is_compiling() else _npu_quant_gmm
    return op(
        x,
        weight,
        scale,
        offset=offset,
        per_token_scale=per_token_scale,
        bias=bias,
        group_list=group_list,
        group_list_type=group_list_type,
        output_dtype=output_dtype,
        act_type=act_type,
    )


def npu_quant_gmm(
    x, weight, scale, *, offset=None, per_token_scale=None, bias=None, group_list=None, output_dtype=None, act_type=0
):
    return _npu_quant_gmm_common(
        x,
        weight,
        scale,
        offset=offset,
        per_token_scale=per_token_scale,
        bias=bias,
        group_list=group_list,
        group_list_type=0,
        output_dtype=output_dtype,
        act_type=act_type,
    )


def npu_quant_gmm_v2(
    x, weight, scale, *, offset=None, per_token_scale=None, bias=None, group_list=None, output_dtype=None, act_type=0
):
    return _npu_quant_gmm_common(
        x,
        weight,
        scale,
        offset=offset,
        per_token_scale=per_token_scale,
        bias=bias,
        group_list=group_list,
        group_list_type=1,
        output_dtype=output_dtype,
        act_type=act_type,
    )
